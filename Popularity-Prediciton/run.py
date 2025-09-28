import argparse
import os
import torch
import numpy as np
import copy
from exp.exp_forecasting import Exp_Forecast

import random
import numpy as np

# 设置随机种子
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# 创建参数解析器
parser = argparse.ArgumentParser(description='TPP-LLM')

# 基础配置
parser.add_argument('--task_name', type=str, required=True, default='popularity_prediction',
                    help='task name, options:[popularity_prediction]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='TPPLLM')

# 数据加载器
parser.add_argument('--data', type=str, required=True, default='mediadata', help='dataset type')
parser.add_argument('--number_variable', type=int, default=7, help='number of variable')
parser.add_argument('--root_path', type=str, default='./data/mediadata', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='weibo.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# 预测任务
parser.add_argument('--seq_len', type=int, default=10, help='input sequence length')
parser.add_argument('--label_len', type=int, default=110, help='start token length')
parser.add_argument('--pred_len', type=int, default=110, help='prediction sequence length')

# 模型定义
parser.add_argument('--top_k', type=int, default=1)
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# 优化
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSLE', help='loss function')
parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--decay_fac', type=float, default=0.75)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=5, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multile gpus')

# 去平稳化投影器参数
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# 补丁设置
parser.add_argument('--patch_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=5)
parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--ln', type=int, default=0)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--weight', type=float, default=0)
parser.add_argument('--percent', type=int, default=5)
parser.add_argument('--pretrained', action='store_false', help='use finetuned GPT2', default=True)

parser.add_argument('--tokenization', type=str, default='patch', help='tokenization_method')
parser.add_argument('--training_strategy', type=str, default='none', help='training_strategy')

parser.add_argument('--add_prompt', type=int, default=0)
parser.add_argument('--add_trainable_prompt', type=int, default=0)
parser.add_argument('--prompt_length', type=int, default=1)
parser.add_argument('--sim_coef', type=float, default=0.0)
parser.add_argument('--pool_size', type=int, default=1000)
parser.add_argument('--period', type=int, default=24)
parser.add_argument('--prompt_init', type=str, default='text_prototype', help='prompt_init_type')
parser.add_argument('--trend_length', type=int, default=24, help='trend_length')
parser.add_argument('--seasonal_length', type=int, default=96, help='seasonal_length')

# 新增参数：patch_size搜索范围
parser.add_argument('--patch_search', type=str, default=None, help='patch_size search range, e.g., "2,4,6,8,10"')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

print('Args in experiment:')
print(args)

# 任务类型对应实验类
if args.task_name == 'popularity_prediction':
    Exp = Exp_Forecast

if args.is_training:
    msles = []

    for ii in range(args.itr):
        # 设置实验记录
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        exp = Exp(args)  # 设置实验

        print('>>>>>>>开始训练 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        best_model_path = path + '/' + 'checkpoint.pth'
        exp.model.load_state_dict(torch.load(best_model_path))

        if args.task_name == 'popularity_prediction':
            msle = exp.test(setting)
            msles.append(msle)
            torch.cuda.empty_cache()
    print('msle_means: ', np.array(msles), 'mean: ', np.mean(np.array(msles)))

else:
    ii = 0
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

    exp = Exp(args)  # 设置实验
    print('>>>>>>>测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()