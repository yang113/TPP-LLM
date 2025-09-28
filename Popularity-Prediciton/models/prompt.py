import torch
import torch.nn as nn

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


class Prompt(nn.Module):
    def __init__(self, length=2, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=30, top_k=1, batchwise_prompt=False, prompt_key_init='uniform', wte=None):
        super().__init__()

        self.length = length  # 1
        self.embed_dim = embed_dim  # 768
        self.prompt_pool = prompt_pool  # false
        self.embedding_key = embedding_key  # mean
        self.prompt_init = prompt_init  # uniform
        self.prompt_key = prompt_key  # true
        self.prompt_key_init = prompt_key_init  # text-prototype
        self.pool_size = pool_size  # 500
        print(self.pool_size)
        self.top_k = top_k  # 4
        self.batchwise_prompt = batchwise_prompt  # false
        self.wte = torch.tensor(torch.load('./word_embedding_representation_500.pt')).to(device)

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(key_shape), requires_grad=False)
                print('zero initialized key')

            elif prompt_key_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(key_shape), requires_grad=False)
                nn.init.uniform_(self.prompt, -5, 5)
                print('uniform initialized key')

            elif prompt_key_init == 'gaussian':
                self.prompt = nn.Parameter(torch.randn(key_shape), requires_grad=False)
                nn.init.normal_(self.prompt, mean=0.0, std=5.0)
                print('gaussian initialized key')

            elif prompt_key_init == 'text_prototype':
                self.text_prototype_linear = nn.Linear(500, pool_size)

        else:

            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_key:  # if self.prompt_pool
            prompt_key = self.wte
            prompt_norm = self.l2_normalize(prompt_key, dim=1)  # Pool_size, C   self.prompt_key

            x_embed_norm = self.l2_normalize(x_embed, dim=2)
            similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B,seq_num,Pool_size

            if prompt_mask is None:
                batch_size, segments, _ = similarity.shape
                idx = []
                for i in range(batch_size):
                    seg_idx = []
                    for j in range(segments):
                        _, top_k_idx = torch.topk(similarity[i, j], k=self.top_k)
                        seg_idx.append(top_k_idx)
                    idx.append(torch.stack(seg_idx))
                idx = torch.stack(idx)
            else:
                idx = prompt_mask  # B, top_k


            batched_prompt_raw = prompt_key[idx]

            batched_prompt_raw = batched_prompt_raw.unsqueeze(3)

            batch_size, segments, top_k, length, c = batched_prompt_raw.shape
            selected_wte = self.wte[idx]
            selected_wte = nn.Parameter(selected_wte, requires_grad=True)
            batched_prompt = selected_wte.reshape(batch_size, segments * top_k * length, c)

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            # prompt_norm = prompt_norm.t()
            batched_key_norm = prompt_norm[idx]
            out['selected_key'] = batched_key_norm
            # x_embed_norm = x_embed_norm.unsqueeze(1)
            x_embed_norm = x_embed_norm.unsqueeze(2)
            sim = batched_key_norm * x_embed_norm
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
            reduce_sim = torch.mean(torch.sum(sim) / (x_embed.shape[0] * self.top_k))

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)


        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = batched_prompt + x_embed
        out['prompt_key'] = prompt_key  # prompt_key

        return out