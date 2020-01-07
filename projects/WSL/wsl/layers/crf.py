from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from wsl import _C


class _CRF(Function):
    @staticmethod
    def forward(
        ctx,
        U,
        I,
        max_iter=10,
        scale_factor=1,
        color_factor=13,
        size_std=500,
        pos_w=3,
        pos_x_std=3,
        pos_y_std=3,
        bi_w=10,
        bi_x_std=80,
        bi_y_std=80,
        bi_r_std=13,
        bi_g_std=13,
        bi_b_std=13,
        debug_info=False,
    ):

        M = _C.crf_forward(
            U,
            I,
            max_iter,
            scale_factor,
            color_factor,
            size_std,
            pos_w,
            pos_x_std,
            pos_y_std,
            bi_w,
            bi_x_std,
            bi_y_std,
            bi_r_std,
            bi_g_std,
            bi_b_std,
            debug_info,
        )

        return M

    @staticmethod
    @once_differentiable
    def backward(ctx, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16):
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


crf = _CRF.apply


class CRF(nn.Module):
    def __init__(
        self,
        max_iter=10,
        scale_factor=1,
        color_factor=13,
        size_std=500,
        pos_w=3,
        pos_x_std=3,
        pos_y_std=3,
        bi_w=10,
        bi_x_std=80,
        bi_y_std=80,
        bi_r_std=13,
        bi_g_std=13,
        bi_b_std=13,
        debug_info=False,
    ):
        super(CRF, self).__init__()

        self.max_iter = max_iter
        self.scale_factor = scale_factor
        self.color_factor = color_factor
        self.size_std = size_std
        self.pos_w = pos_w
        self.pos_x_std = pos_x_std
        self.pos_y_std = pos_y_std
        self.bi_w = bi_w
        self.bi_x_std = bi_x_std
        self.bi_y_std = bi_y_std
        self.bi_r_std = bi_r_std
        self.bi_g_std = bi_g_std
        self.bi_b_std = bi_b_std
        self.debug_info = debug_info

    def forward(self, U, I):
        return crf(
            U,
            I,
            self.max_iter,
            self.scale_factor,
            self.color_factor,
            self.size_std,
            self.pos_w,
            self.pos_x_std,
            self.pos_y_std,
            self.bi_w,
            self.bi_x_std,
            self.bi_y_std,
            self.bi_r_std,
            self.bi_g_std,
            self.bi_b_std,
            self.debug_info,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "max_iter=" + str(self.max_iter)
        tmpstr += "scale_factor=" + str(self.scale_factor)
        tmpstr += "color_factor=" + str(self.color_factor)
        tmpstr += "size_std=" + str(self.size_std)
        tmpstr += "pos_w=" + str(self.pos_w)
        tmpstr += "pos_x_std=" + str(self.pos_x_std)
        tmpstr += "pos_y_std=" + str(self.pos_y_std)
        tmpstr += "bi_w=" + str(self.bi_w)
        tmpstr += "bi_x_std=" + str(self.bi_x_std)
        tmpstr += "bi_y_std=" + str(self.bi_y_std)
        tmpstr += "bi_r_std=" + str(self.bi_r_std)
        tmpstr += "bi_g_std=" + str(self.bi_g_std)
        tmpstr += "bi_b_std=" + str(self.bi_b_std)
        tmpstr += "debug_info=" + str(self.debug_info)
        tmpstr += ")"
        return tmpstr
