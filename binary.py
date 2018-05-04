import math
import torch

import numpy as np
import torch.nn as nn

from torch.nn import functional as F
from torch.autograd import Variable
from torch.autograd import Function


class BinaryLinearFunction(Function):

    @staticmethod
    def forward(ctx, input_tensor, w, b=None):
        ctx.save_for_backward(input_tensor, w, b)

        input_tensor_b = torch.sign(input_tensor)
        #Binarize Params
        w_b = torch.sign(w)
        if not b is None:
            b_b = torch.sign(b)

        #Do Multiplicaiton
        output = input_tensor_b.mm(w_b.t())
        if not b is None:
            output += b_b.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, w, b = ctx.saved_variables

        grad_input = grad_output.mm(w)
        grad_weight = grad_output.t().mm(input_tensor)
        grad_bias = grad_output.sum(0).squeeze(0)

        if not b is None:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight, None


class ClippedBinaryLinearFunction(Function):

    @staticmethod
    def forward(ctx, input_tensor, w, b=None):
        ctx.save_for_backward(input_tensor, w, b)

        input_tensor_b = torch.sign(input_tensor)
        #Binarize Params
        w_b = torch.sign(w)
        if not b is None:
            b_b = torch.sign(b)

        #Do Multiplicaiton
        output = input_tensor_b.mm(w_b.t())
        if not b is None:
            output += b_b.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        #TODO implement clipping
        input_tensor, w, b = ctx.saved_variables

        grad_input = grad_output.mm(w)
        grad_weight = grad_output.t().mm(input_tensor)
        grad_weight[grad_weight > 0] = 0
        #grad_weight = grad_weight.clamp(min=-1, max=1)
        grad_bias = grad_output.sum(0).squeeze(0)
        grad_bias[grad_bias > 0] = 0
        #grad_bias = grad_bias.clamp(min=-1, max=1)

        if not b is None:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight, None


class BinaryLinearUnit(nn.Module):

    def __init__ (self,
                  in_features,
                  out_features,
                  use_bias=False,
                  clip_grads=False):

        super(BinaryLinearUnit, self).__init__()

        self._w = nn.Parameter(torch.Tensor(out_features, in_features))
        self._b = None
        if use_bias:
            self._b = nn.Parameter(torch.Tensor(out_features, ))
        self.init_params()

        if clip_grads:
            self._binary_linear_f = ClippedBinaryLinearFunction.apply
        else:
            self._binary_linear_f = BinaryLinearFunction.apply

        self._batchnorm_f = nn.BatchNorm1d(out_features)


    def init_params(self):
        stdv = 1. / math.sqrt(self._w.size(1))
        self._w.data.uniform_(-stdv, stdv)
        if not self._b is None:
            self._b.data.uniform_(-stdv, stdv)
        return


    def forward(self, input_var):
        x = self._binary_linear_f(input_var, self._w, self._b)
        x = self._batchnorm_f(x)
        return x


    def clip(self):
        self._w.data = np.clip(self._w.data, -1, 1)
        if not self._b is None:
            self._b.data = np.clip(self._b.data, -1, 1)
        return


class BinaryNet(nn.Module):

    def __init__(self,
                 use_dropout=False,
                 use_grad_clip=False):
        super(BinaryNet, self).__init__()

        self.use_dropout = use_dropout

        self.inbn = nn.BatchNorm1d(784)
        self.bfc1 = BinaryLinearUnit(784,
                                     2048,
                                     use_bias=True,
                                     clip_grads=use_grad_clip)
        self.dropout1 = nn.Dropout(.2)
        self.bfc2 = BinaryLinearUnit(2048,
                                     2048,
                                     use_bias=True,
                                     clip_grads=use_grad_clip)
        self.dropout2 = nn.Dropout(.2)
        self.bfc3 = BinaryLinearUnit(2048,
                                     10,
                                     use_bias=True,
                                     clip_grads=use_grad_clip)

        self.smw = nn.Linear(10, 10)
        self.sm =  nn.Softmax()

    def forward(self, input_var):
        x = self.inbn(input_var)
        x = self.bfc1(x)

        if self.use_dropout:
            x = self.dropout1(x)

        x = self.bfc2(x)

        if self.use_dropout:
            x = self.dropout2(x)

        x = self.bfc3(x)
        x = self.smw(x)
        x = self.sm(x)
        return x


    def clip(self):
        self.bfc1.clip()
        self.bfc2.clip()
        self.bfc3.clip()


class BigBinaryNet(nn.Module):

    def __init__(self,
                 use_dropout=False,
                 use_grad_clip=False):
        super(BigBinaryNet, self).__init__()

        self.use_dropout = use_dropout

        self.inbn = nn.BatchNorm1d(784)
        self.bfc1 = BinaryLinearUnit(784,
                                     4096,
                                     use_bias=True,
                                     clip_grads=use_grad_clip)
        self.dropout1 = nn.Dropout()
        self.bfc2 = BinaryLinearUnit(4096,
                                     4096,
                                     use_bias=True,
                                     clip_grads=use_grad_clip)
        self.dropout2 = nn.Dropout()
        self.bfc3 = BinaryLinearUnit(4096,
                                     10,
                                     use_bias=True,
                                     clip_grads=use_grad_clip)

        self.smw = nn.Linear(10, 10)
        self.sm =  nn.Softmax()

    def forward(self, input_var):
        x = self.inbn(input_var)
        x = self.bfc1(x)

        if self.use_dropout:
            x = self.dropout1(x)

        x = self.bfc2(x)

        if self.use_dropout:
            x = self.dropout2(x)

        x = self.bfc3(x)
        x = self.smw(x)
        x = self.sm(x)
        return x


    def clip(self):
        self.bfc1.clip()
        self.bfc2.clip()
        self.bfc3.clip()
