import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from wsl import _C


class _CSC(Function):
    @staticmethod
    def forward(
        ctx,
        cpgs,
        labels,
        preds,
        rois,
        tau,
        debug_info,
        fg_threshold,
        mass_threshold,
        density_threshold,
        area_sqrt,
        context_scale,
    ):
        PL = labels.clone().detach()
        NL = torch.zeros(labels.size(), dtype=labels.dtype, device=labels.device)

        W = _C.csc_forward(
            cpgs,
            labels,
            preds,
            rois,
            tau,
            debug_info,
            fg_threshold,
            mass_threshold,
            density_threshold,
            area_sqrt,
            context_scale,
        )

        return W, PL, NL

    @staticmethod
    @once_differentiable
    def backward(ctx, dW, dPL, dNL):
        return None, None, None, None


csc = _CSC.apply


class CSC(nn.Module):
    def __init__(
        self,
        tau=0.7,
        debug_info=False,
        fg_threshold=0.1,
        mass_threshold=0.2,
        density_threshold=0.0,
        area_sqrt=True,
        context_scale=1.8,
    ):
        super(CSC, self).__init__()

        self.tau = tau
        self.debug_info = debug_info
        self.fg_threshold = fg_threshold
        self.mass_threshold = mass_threshold
        self.density_threshold = density_threshold
        self.area_sqrt = area_sqrt
        self.context_scale = context_scale

    def forward(self, cpgs, labels, preds, rois):
        return csc(
            cpgs,
            labels,
            preds,
            rois,
            self.tau,
            self.debug_info,
            self.fg_threshold,
            self.mass_threshold,
            self.density_threshold,
            self.area_sqrt,
            self.context_scale,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "tau=" + str(self.tau)
        tmpstr += ", debug_info=" + str(self.debug_info)
        tmpstr += ", fg_threshold=" + str(self.fg_threshold)
        tmpstr += ", mass_threshold=" + str(self.mass_threshold)
        tmpstr += ", density_threshold=" + str(self.density_threshold)
        tmpstr += ", area_sqrt=" + str(self.area_sqrt)
        tmpstr += ", context_scale=" + str(self.context_scale)
        tmpstr += ")"
        return tmpstr


class _CSCConstraint(Function):
    @staticmethod
    def forward(ctx, X, W, polar):

        if polar:
            W_ = torch.clamp(W, min=0.0)
        else:
            W_ = torch.clamp(W, max=0.0)
            W_ = W_ * (-1.0)

        ctx.save_for_backward(W_)

        Y = X * W_

        return Y

    @staticmethod
    @once_differentiable
    def backward(ctx, dY):
        (W_,) = ctx.saved_tensors

        dX = dY * W_

        return dX, None, None


csc_constraint = _CSCConstraint.apply


class CSCConstraint(nn.Module):
    def __init__(self, polar=True):
        super(CSCConstraint, self).__init__()

        self.polar = polar

    def forward(self, X, W):
        return csc_constraint(X, W, self.polar)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "polar=" + str(self.polar)
        tmpstr += ")"
        return tmpstr
