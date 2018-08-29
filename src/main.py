"""Main pipeline of DA-RNN.

@author Zhenye Na 05/21/2018

"""

import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
# import matplotlib.pyplot as plt

# from tqdm import tqdm
from torch.autograd import Variable

from ops import *
from model import *

# Parameters settings
parser = argparse.ArgumentParser(description="DA-RNN")

# Dataset setting
parser.add_argument(
    '--dataroot',
    type=str,
    default="../nasdaq/nasdaq100_padding.csv",
    help='path to dataset')
parser.add_argument(
    '--workers', type=int, default=2, help='number of data loading workers [2]')
parser.add_argument(
    '--batchsize', type=int, default=128, help='input batch size [128]')

# Encoder / Decoder parameters setting
parser.add_argument(
    '--nhidden_encoder',
    type=int,
    default=128,
    help='size of hidden states for the encoder m [64, 128]')
parser.add_argument(
    '--nhidden_decoder',
    type=int,
    default=128,
    help='size of hidden states for the decoder p [64, 128]')
parser.add_argument(
    '--ntimestep',
    type=int,
    default=10,
    help='the number of time steps in the window T [10]')

# Training parameters setting
parser.add_argument(
    '--epochs',
    type=int,
    default=80,
    help='number of epochs to train [10, 200, 500]')
parser.add_argument(
    '--resume', type=bool, default=False, help='resume training or not')
parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args('--manualSeed=1 '.split())

torch.manual_seed(1)
# Read dataset
X, y = read_data(opt.dataroot, debug=False)

# Initialize model
model = DA_rnn(X, y, opt.ntimestep, opt.nhidden_encoder, opt.nhidden_decoder,
               opt.batchsize, opt.lr, opt.epochs)

# Train
model.train()

# Prediction
y_pred = model.test()

# fig1 = plt.figure()
# plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
# plt.savefig("1.png")
# plt.close(fig1)

# fig2 = plt.figure()
# plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
# plt.savefig("2.png")
# plt.close(fig2)

# fig3 = plt.figure()
# plt.plot(y_pred, label='Predicted')
# plt.plot(model.y[model.train_timesteps:], label="True")
# plt.legend(loc='upper left')
# plt.savefig("3.png")
# plt.close(fig3)
# print('Finished Training')

np.save('ypred3_2', y_pred.detach().numpy())
a=np.load('ypred3.npy')
aa=np.load('ypred3_2.npy')
print((a==aa).sum())

np.save('weight1',self.encoder_attn[0].weight.detach().numpy())
np.save('weight1_1',self.encoder_attn[0].weight.detach().numpy())
a=np.load('weight1.npy')
aa=np.load('weight1_1.npy')
assert (a==aa).sum()==a.shape[0]*a.shape[1]

np.save('weight0',self.encoder_attn[0].weight.detach().numpy())
np.save('weight0_1',self.encoder_attn[0].weight.detach().numpy())
a=np.load('weight0.npy')
aa=np.load('weight0_1.npy')
assert (a==aa).sum()==a.shape[0]*a.shape[1]

# # test
# x=torch.arange(0,32)
# x=x.view(-1,4,4)
# w=torch.ones(1,4)
# r=F.linear(x,w)
# print(r.squeeze(2))

# x=x.view(-1,4)
# rr=F.linear(x,w)
# print(rr.view(-1,4))
