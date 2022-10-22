# 220609

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

from .prep import padding, smiles_vocab_dict, char_to_idx
from .models import Seq2Seq 

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

class LearningRateScheduler():
    def __init__(self,base_lr,max_epoch,power):
        self.max_epoch = max_epoch
        self.power = power
        self.base_lr = base_lr

    def __call__(self,epoch):
        return (1 - max(epoch - 1, 1) / self.max_epoch) ** self.power

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train2batch(data,batch_size=1024,shuffle_data=True):
    # data: list of training data array
    batch = []
    if shuffle_data == True:
        data = shuffle(*data)
    check = len(data[0]) // batch_size
    for i in np.arange(0,check*batch_size+1,batch_size):
        if i == len(data[0]):
            continue
        elif i + batch_size <= len(data[0]):
            temp = [v[i:i+batch_size] for v in data]
        else:
            temp = [v[i:] for v in data]
        batch.append(temp)
    return batch

def get_max_index(decoder_out):
    res = []
    dec_out = decoder_out.transpose(1,0)
    for h in dec_out:
        res.append(torch.argmax(h))
    return torch.tensor(res,device=DEVICE).view(1,dec_out.size(0))


class Trainer():
    def __init__(self,smiles_input,smiles_output,seed=42):
        self.input = smiles_input
        self.output = smiles_output
        self.vocab = smiles_vocab_dict()
        self.id2sm = {w:v for v,w in self.vocab.items()}
        fix_seed(seed)

    def tokenize(self):
        i, _ = char_to_idx(self.input,self.vocab)
        o, _ = char_to_idx(self.output,self.vocab)
        self.input_token = i
        self.output_token = o

    def bucketing(self,min,max,step_width,test_size):
        train_i, test_i, train_o, test_o = train_test_split(self.input_token,self.output_token,
                                                            test_size=test_size,random_state=0)

        proc = pd.DataFrame({"input":train_i,"output":train_o,"length":[len(v) for v in train_i]})
        bucket_step = np.arange(min,max+1,step_width)
        buckets = []
        for i,v in enumerate(bucket_step):
            if i == 0:
                temp = proc[proc["length"] < v]
            else:
                temp = proc[(proc["length"] >= bucket_step[i-1]) & (proc["length"] < v)]
            buckets.append(temp)
        buckets.append(proc[proc["length"] >= bucket_step[-1]])

        self.buckets = []
        for i,v in enumerate(buckets):
            maxlen = np.max(v["length"])
            temp = padding(list(v["input"]),maxlen).astype(np.int64)
            maxlen2 = np.max([len(w) for w in v["output"]])
            temp2 = padding(list(v["output"]),maxlen2).astype(np.int64)
            self.buckets.append([temp,temp2])

        self.tests = [test_i,test_o]

    def train(self,encoder,decoder,criterion,optimizer,scheduler=None,epochs=1000,batch_size=1024,
              loss_plot=True,progress=False):
        self.model = Seq2Seq(encoder,decoder).to(DEVICE)
        self.batch_size = batch_size

        self.model.train()
        self.loss = []
        for epoch in tqdm(range(epochs)):
            buc = epoch % len(self.buckets)
            batch = train2batch(self.buckets[buc],batch_size=batch_size)
            epoch_loss = self._train(optimizer,batch,criterion)
            
            if scheduler != None:
                scheduler.step()
            self.loss.append(epoch_loss)
            if progress == True:
                if epoch % len(self.buckets) == len(self.buckets) - 1:
                    loss_sum = np.round(np.sum(self.loss[-len(self.buckets):]),2)
                    print("loss {}-{} : {}".format(epoch+1-len(self.buckets),epoch,loss_sum))

        if loss_plot:
            loss_ = []
            l = 0
            for i,v in enumerate(self.loss):
                l += v
                if i % len(self.buckets) == len(self.buckets) - 1:
                    loss_.append(l)
                    l = 0
            self.loss_ = loss_
            plt.plot(np.arange(1,len(loss_)+1,1),loss_)

    def _train(self,optimizer,batch,criterion):
        epoch_loss = 0
        for v in batch:
            optimizer.zero_grad()
            inputs = torch.tensor(v[0],device=DEVICE,dtype=torch.long).t()
            outputs = torch.tensor(v[1],device=DEVICE,dtype=torch.long).t()

            target = outputs[1:,:]
            source = outputs[:-1,:]
            dec_out, _ = self.model(inputs,source)
            l = 0
            for j in range(target.size(0)):
                l += criterion(dec_out[j],target[j])
            epoch_loss += l.item()
            l.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1)
            optimizer.step()

        return epoch_loss

    def inference(self):
        self.model.eval()
        test_i = padding(self.tests[0],np.max([len(v) for v in self.tests[0]])).astype(np.int64)
        test_o = padding(self.tests[1],np.max([len(v) for v in self.tests[1]])).astype(np.int64)
        self.test_batch = train2batch([test_i,test_o],batch_size=self.batch_size)
        self.predicts = []  
        with torch.no_grad():      
            for i,v in enumerate(self.test_batch):
                input_tensor = torch.tensor(v[0],device=DEVICE,dtype=torch.long).t()
                _, _, encoder_state = self.model.encoder(input_tensor,inference=True) 

                start_char_batch = [1 for _ in range(input_tensor.size(1))] 
                dec_input_tensor = torch.tensor(start_char_batch,device=DEVICE).unsqueeze(0) # [1, B]
                dec_hid = encoder_state #[B, Hsum]

                batch_tmp = torch.zeros(1,dec_input_tensor.size(1),dtype=torch.long,device=DEVICE)
                for _ in range(input_tensor.size(0)): # teacher forcing
                    dec_o, dec_hid = self.model.decoder(dec_input_tensor,dec_hid)
                    dec_hid = torch.cat(dec_hid,axis=1)
                    dec_input_tensor = get_max_index(dec_o)
                    batch_tmp = torch.cat([batch_tmp,dec_input_tensor],axis=0)
                
                self.predicts.append(batch_tmp[1:,:])

    def evaluate(self):
        row = []
        for s,t in zip(self.test_batch,self.predicts):
            for inp, out, pred in zip(s[0],s[1],t.T): # [B, T]
                x = [self.id2sm[j] for j in inp]
                y = [self.id2sm[j] for j in out]
                p = [self.id2sm[j] for j in pred[0]] if self.beam_width > 1 else [self.id2sm[j.item()] for j in pred]
                
                x_str = "".join(x[1:]).split(self.id2sm[2])[0]
                y_str = "".join(y[1:]).split(self.id2sm[2])[0]
                p_str = "".join(p).split(self.id2sm[2])[0]

                judge = "O" if y_str == p_str else "X"
                row.append([x_str,y_str,p_str,judge])
        self.pred_df = pd.DataFrame(row,columns=["input","answer","predict","judge"])
        self.accuracy = len(self.pred_df.query("judge == 'O'")) / len(self.pred_df)

        return self.pred_df, self.accuracy

    def generate(self,state,maxlen=1000,batch=1024):
        self.model.eval()
        with torch.no_grad():
            self.generates = []
            if type(state).__module__ == "numpy":
                state = torch.tensor(state,device=DEVICE,dtype=torch.float)
            state = train2batch([state],batch_size=batch,shuffle_data=False)
            for s in state:
                enc_state = self.model.encoder.linear2(s[0])

                start_char_batch = [1 for _ in range(s[0].size(0))] 
                dec_input_tensor = torch.tensor(start_char_batch,device=DEVICE,dtype=torch.long).unsqueeze(0) # [1, B]
                dec_hid = enc_state 
                batch_tmp = torch.zeros(1,dec_input_tensor.size(1),dtype=torch.long,device=DEVICE)
                fin_set = set()
                i = 0
                while i < maxlen:
                    dec_o, dec_hid = self.model.decoder(dec_input_tensor,dec_hid)
                    dec_hid = torch.cat(dec_hid,axis=1)
                    dec_input_tensor = get_max_index(dec_o)
                    batch_tmp = torch.cat([batch_tmp,dec_input_tensor],axis=0)
                    fin_set = fin_set | set((dec_input_tensor==2).nonzero(as_tuple=True)[1].cpu().detach().numpy())
                    fin_set = fin_set | set((dec_input_tensor==0).nonzero(as_tuple=True)[1].cpu().detach().numpy())
                    if len(fin_set) == s[0].size(0):
                        break
                    i += 1
                self.generates.append(batch_tmp[1:,:])
        
        row = []
        for v in self.generates:
            for s in v.T:
                p = [self.id2sm[j.item()] for j in s]
                p_str = "".join(p).split(self.id2sm[2])[0]
                row.append(p_str)

        return row

    def get_descriptor(self,smiles,batch=1024):
        self.model.eval()
        i, l = char_to_idx(smiles,self.vocab)
        temp = padding(i,l).astype(np.int64)
        temp = train2batch([temp],batch_size=batch,shuffle_data=False)
        result = []
        with torch.no_grad():
            for v in temp:
                t = torch.tensor(v[0],device="cuda",dtype=torch.long).t()
                _, cmap_desc, _ = self.model.encoder(t,inference=True)
                result.append(cmap_desc.cpu().detach().numpy())
        return np.concatenate(result)

    def save(self,path="trained_params.pth"):
        torch.save(self.model.state_dict(),path)

    def load(self,encoder,decoder,path="trained_params.pth"):
        self.model = Seq2Seq(encoder,decoder)
        self.model.load_state_dict(torch.load(path))