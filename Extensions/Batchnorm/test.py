
import torch as th
import batchnorm_cpp as bn
import numpy as np
# from batchnorm_ext import BatchNorm
import torch.nn as nn

import time
# from batchnorm_ext import batchnorm



#x = th.randint(1, 10, [8, 10, 3])
#x = th.randn([5,6,4])
# x = th.randn([1, 3, 16])
#print(th.mean(x))
# print("x: ", x)

# print("Numpy Mean")

# y=th.Tensor.cpu(x).detach().numpy()[:,:,:]
# mu = y.mean(axis=(1,2))
# print(mu.shape)
# print(mu)

# x_hat = th.randint(1,10, [100, 16])
x_hat = th.randn([1000000, 1600])
print(x_hat.shape)

#use_gpu = 0
#with th.autograd.profiler.profile(args.enable_profiling, use_gpu, True) as prof:
t0 = time.time()
#for i  in range(300):
bnx_nn = nn.BatchNorm1d(1600) 
nx = bnx_nn(x_hat)
print('Pytorch FWD BN time: %.7f' %(time.time() - t0))

print("BN_NN:", nx)

t1 = time.time()
#for i  in range(300):
gamma= th.ones([1])
beta = th.zeros([1])
# bmean = []
# bavar = []
inp_nrm = th.empty(x_hat.shape)
# bnc = BatchNorm(16)
# bnx = bnc(x_hat)
bnx, bmean, bvar, inp_nrm = bn.batchnorm_fwd_impl(x_hat, gamma, beta)
# bnx = bn.batchnorm_fwd_impl(x_hat, gamma, beta)
# bn.batchnorm_fwd_impl(x_hat, gamma, beta)

# model = batchnorm(16)
# bnx = model(x_hat)

print('Pytorch Extension FWD BN time: %.7f' %(time.time() - t1))

print(bnx)
# print(dgm)
# print(dbt)
# print("inp_nrm: ", inp_nrm)