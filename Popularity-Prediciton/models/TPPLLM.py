import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
from models.mlp import MLP

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from transformers import GPT2Tokenizer
from models.Cross_Modal_Align import CrossModal
from models.StandardNorm import Normalize
from models.prompt import Prompt

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):

    def __init__(self, configs):  # 添加use_prompt_pool参数
        super(Model, self).__init__()
        self.configs = configs
        self.use_prompt_pool = True  # 控制是否使用提示池
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.e_layer = 1
        self.d_ff = 768
        self.channel = 768
        self.head = 8
        self.dropout_n = 0.5
        self.num_nodes = 768
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.relu = nn.ReLU()
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(device)

        self.tokenizer = GPT2Tokenizer.from_pretrained('./gpt2')
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(device)

        # Time Series Encoder
        self.ts_encoder_layer = nn.TransformerEncoderLayer(d_model=self.channel, nhead=self.head, batch_first=True,
                                                           norm_first=True, dropout=self.dropout_n).to(device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers=self.e_layer).to(device)

        # Prompt Encoder
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_ff, nhead=self.head, batch_first=True,
                                                               norm_first=True, dropout=self.dropout_n).to(device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers=self.e_layer).to(device)

        # Cross-modality alignment
        self.cross = CrossModal(d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
                                attn_dropout=self.dropout_n,
                                dropout=self.dropout_n, pre_norm=True, activation="relu", res_attention=True,
                                n_layers=1, store_attn=False).to(device)

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.channel, nhead=self.head, batch_first=True,
                                                        norm_first=True, dropout=self.dropout_n).to(device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2).to(device)

        # Projection
        self.c_to_length = nn.Sequential(
            nn.Linear(self.channel, self.pred_len, bias=True).to(device),
            nn.Softplus()
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * 768, 768),
            nn.LayerNorm(768),
            nn.GELU()
        )

        # GPT2模型
        if configs.pretrained == True:
            self.gpt2 = GPT2Model.from_pretrained('./gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        else:
            print("------------------no pretrain------------------")
            self.gpt2 = GPT2Model(GPT2Config())

        # 仅在使用提示池时初始化提示池模块
        if self.use_prompt_pool:
            self.prompt_pool = Prompt(
                length=1,
                embed_dim=768,
                embedding_key='mean',
                prompt_init='uniform',
                prompt_pool=False,
                prompt_key=True,
                pool_size=self.configs.pool_size,
                top_k=self.configs.prompt_length,
                batchwise_prompt=False,
                prompt_key_init=self.configs.prompt_init,
                wte=self.gpt2.wte.weight
            ).to(device)
        else:
            self.prompt_pool = None  # 不使用提示池时设为None

        # GPT2参数冻结设置
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x_enc, train_embedding, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out, res = self.forecast(x_enc, train_embedding)
            return dec_out[:, -self.pred_len:, :], res  # [B, L, D]
        return None

    def forecast(self, x_enc, train_embedding):
        B, L, M = x_enc.shape  # (64,5,1)
        x_enc = x_enc.float()
        train_embedding = train_embedding.float()  # (64,768,1)

        x_enc = x_enc.permute(0, 2, 1)  # (64,1,5)
        x_enc = self.length_to_feature(x_enc)  # (64,1,768)

        train_embedding = train_embedding.permute(0, 2, 1)  # [B, N, E] (64,1,768)

        # 编码器处理
        enc_out = self.ts_encoder(x_enc)  # [64, 1, 768]
        embeddings = self.prompt_encoder(train_embedding)  # [64, 1, 768]

        # 跨模态对齐
        cross_out = self.cross(embeddings, enc_out, enc_out)  # [B, C, N]
        # cross_out = enc_out + embeddings
        cross_out = self.cross(cross_out, embeddings, embeddings)
        # concat = torch.cat([enc_out, embeddings], dim=2)
        # cross_out = self.projection(concat)

        # 提示池处理分支：根据use_prompt_pool决定是否使用提示池
        if self.use_prompt_pool and self.prompt_pool is not None:
            prompt_similar = self.prompt_pool(cross_out)
            prompt_similar_embeddings = prompt_similar['prompted_embedding']
            similar_loss = prompt_similar['reduce_sim']
        else:
            # 不使用提示池时，直接使用cross_out作为嵌入
            prompt_similar_embeddings = cross_out
            similar_loss = torch.tensor(0.0, device=device)  # 无相似性损失

        res = {'similarity_loss': similar_loss}

        # GPT2处理
        last_embedding = self.gpt2(inputs_embeds=prompt_similar_embeddings).last_hidden_state

        # 输出投影
        outputs = self.c_to_length(last_embedding.reshape(B * M * 1, -1))
        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B, m=M, c=1)
        outputs = outputs.sum(dim=2)
        outputs = rearrange(outputs, 'b m l -> b l m')

        return outputs, res