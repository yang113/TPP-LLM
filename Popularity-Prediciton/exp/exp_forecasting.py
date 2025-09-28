from statsmodels.multivariate.factor_rotation import promax

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,adjust_model
from utils.metrics import metric
import torch
import torch.nn as nn
from models import  TPPLLM
from torch.nn.utils import clip_grad_norm_
from utils.losses import mape_loss, msle_loss


from transformers import AdamW





from torch.utils.data import Dataset, DataLoader
from torch import optim
import os
import time
import warnings
import numpy as np

from tqdm import tqdm

warnings.filterwarnings('ignore')


class Exp_Forecast(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TPP-LLM': TPPLLM,
            
        }

        self.device = torch.device('cuda:6')
        self.model = self._build_model()
        
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        # self.test_data, self.test_loader = self._get_data(flag='test')

        self.optimizer = self._select_optimizer()    # Adam
        self.criterion = self._select_criterion()    # msle_loss

      

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = msle_loss()

        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, embeddings) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                valid_embedding = torch.Tensor(embeddings).to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,res = self.model(batch_x,valid_embedding)
                        else:
                            outputs,res = self.model(batch_x, valid_embedding)
                else:
                    if self.args.output_attention:
                        outputs,res = self.model(batch_x, valid_embedding)
                    else:
                        outputs,res = self.model(batch_x, valid_embedding)
                f_dim = 0

                outputs = outputs[:, :self.args.pred_len, f_dim:self.args.number_variable].float().to(self.device)
                batch_y = batch_y[:, :self.args.label_len, f_dim:self.args.number_variable].float().to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    

   

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y,embeddings) in tqdm(enumerate(self.train_loader)):
                iter_count += 1
                self.optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                train_embedding = torch.Tensor(embeddings).to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,res = self.model(batch_x, train_embedding)

                        else:
                            outputs,res = self.model(batch_x, train_embedding)


                        f_dim=0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                        loss = self.criterion(outputs, batch_y)

    
                else:
                    if self.args.output_attention:
                        outputs,res = self.model(batch_x, train_embedding)

                    else:
                        outputs,res = self.model(batch_x, train_embedding)



                    f_dim = 0
                    outputs = outputs[:, :self.args.pred_len, f_dim:self.args.number_variable]
                    batch_y = batch_y[:, :self.args.pred_len, f_dim:self.args.number_variable].float().to(self.device)
                    loss = self.criterion(outputs, batch_y)
                    
                    train_loss.append(loss.item())


                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    loss.backward()
                    self.optimizer.step()
                else:
                 
                    loss.backward()
                    self.optimizer.step()
               
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.vali_data, self.vali_loader, self.criterion)
    

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
            adjust_model(self.model, epoch + 1,self.args)


            

    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
      
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, embeddings) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                test_embedding = torch.Tensor(embeddings).to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,res = self.model(batch_x, test_embedding)

                        else:

                            outputs, res = self.model(batch_x, test_embedding)
                else:
                    if self.args.output_attention:
                        outputs, res =  self.model(batch_x, test_embedding)

                    else:
                        outputs, res =  self.model(batch_x, test_embedding)

                f_dim = 0


                outputs_paint = outputs[:, :self.args.pred_len, f_dim:self.args.number_variable].float().detach().cpu().numpy()
                batch_y_paint = batch_y[:, :self.args.pred_len, f_dim:self.args.number_variable].float().detach().cpu().numpy()
                outputs_paint_log = np.log(outputs_paint)
                batch_y_paint_log = np.log(batch_y_paint)
                outputs = outputs[:, -1:, f_dim:self.args.number_variable].float().detach().cpu().numpy()
                batch_y = batch_y[:, -1:, f_dim:self.args.number_variable].float().detach().cpu().numpy()

                pred = outputs
                true = batch_y


                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.float().detach().cpu().numpy()
                    input_log = np.log(input)
                    gt = np.concatenate((input_log[0, :, -1], batch_y_paint_log[0, :, -1]), axis=0)
                    pd = np.concatenate((input_log[0, :, -1], outputs_paint_log[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        
       
  
            
        preds = np.array(preds)
        trues = np.array(trues)
            
            
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        msle = metric(preds, trues)
        print('msle:{}'.format(msle))
        f = open("result_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('msle:{}'.format(msle))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([msle]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return msle
    
