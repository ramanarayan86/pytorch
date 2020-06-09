#!/usr/bin/python

import torch
import embedding_cpp as EmbedExt
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function

from torch import Tensor
# from .module import Module
# from .. import functional as F
# from .. import init


class EmbeddingExtFunction(Function):    
    @staticmethod
    def forward(ctx, weight, indices):
        result = EmbedExt.embedding_forward(weight, indices)
        # print("gamma fwd:", gamma)
        # print("beta:", beta)
        ctx.save_for_backward(weight, indices)
        return result
        
    @staticmethod
    def backward(ctx, grad):
        r = ctx.saved_tensors
        weight = r[0]
        indices = r[1]
        grad_out = EmbedExt.embedding_backward(grad, weight, indices)
        return grad_out, None 

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False):
    return EmbeddingExtFunction.apply(weight, input)
   
class EmbeddingExt(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(EmbeddingExt, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)
        self.sparse = sparse
    

    def reset_parameters(self):
        init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        # self._check_input_dim(input)
        return embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)



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
