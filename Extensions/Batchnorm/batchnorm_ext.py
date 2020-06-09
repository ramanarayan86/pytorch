#!/usr/bin/python

import torch
import batchnorm_cpp as bn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function



class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_NormBase, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)



class BatchnormFunction(Function):    
    @staticmethod
    def forward(ctx, input, gamma, beta):
        result, inp_nrm, bmean, bvar = bn.batchnorm_fwd_impl(input, gamma, beta)
        # print("gamma fwd:", gamma)
        # print("beta:", beta)
        # print("input fwd:", input)
        ctx.backward_cache = bmean, bvar, inp_nrm
        ctx.save_for_backward(input, gamma, beta)
        return result
        
    @staticmethod
    def backward(ctx, grad):
        mean, var, inp_nrm = ctx.backward_cache
        r = ctx.saved_tensors
        inp= r[0]
        gamma = r[1]
        beta = r[2]
        # print("gamma:", gamma)
        # print("mean:", mean)
        # print("beta:", beta)
        result, dgamma, dbeta = bn.batchnorm_bwd_impl(grad, inp, inp_nrm, mean, var, gamma, beta)

        return result, dgamma, dbeta 


class BatchNorm(_NormBase):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=False):
        super(BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        # self._check_input_dim(input)
        return BatchnormFunction.apply(input, self.weight, self.bias)



# if __name__ == "__main__":
#     model = dropout(0.5)
#     ten = th.ones([18,18])
#     result = d.dropout_impl(ten, 0.5, True)
#     print(ten)
#     print(result)
#     print()
#     result = model.apply(th.ones([4,4]))
#     #result = model(th.ones([4,4]))
#     print(result)
