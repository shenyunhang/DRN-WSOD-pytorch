import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from wsl import _C


class _PCLLoss(Function):
    @staticmethod
    def forward(
        ctx,
        pcl_probs,
        labels,
        cls_loss_weights,
        gt_assignment,
        pc_labels,
        pc_probs,
        pc_count,
        img_cls_loss_weights,
        im_labels,
    ):
        device_id = pcl_probs.get_device()
        pcl_probs = pcl_probs.data.cpu()
        ctx.save_for_backward(
            pcl_probs,
            labels,
            cls_loss_weights,
            gt_assignment,
            pc_labels,
            pc_probs,
            pc_count,
            img_cls_loss_weights,
            im_labels,
            torch.tensor(device_id),
        )

        output = pcl_probs.new(1, pcl_probs.shape[1]).zero_()

        _C.pcl_loss_forward(
            pcl_probs,
            labels,
            cls_loss_weights,
            pc_labels,
            pc_probs,
            img_cls_loss_weights,
            im_labels,
            output,
        )

        return output.cuda(device_id).sum() / pcl_probs.size(0)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            pcl_probs,
            labels,
            cls_loss_weights,
            gt_assignment,
            pc_labels,
            pc_probs,
            pc_count,
            img_cls_loss_weights,
            im_labels,
            device_id,
        ) = ctx.saved_tensors

        if grad_output.is_cuda:
            grad_output = grad_output.data.cpu()

        grad_input = grad_output.new(pcl_probs.size()).zero_()

        _C.pcl_loss_backward(
            pcl_probs,
            labels,
            cls_loss_weights,
            gt_assignment,
            pc_labels,
            pc_probs,
            pc_count,
            img_cls_loss_weights,
            im_labels,
            grad_output,
            grad_input,
        )

        grad_input /= pcl_probs.size(0)

        return grad_input.cuda(device_id.item()), None, None, None, None, None, None, None, None


pcl_loss = _PCLLoss.apply


class PCLLoss(nn.Module):
    def forward(
        self,
        pcl_prob,
        labels,
        cls_loss_weights,
        gt_assignment,
        pc_labels,
        pc_probs,
        pc_count,
        img_cls_loss_weights,
        im_labels_real,
    ):
        return pcl_loss(
            pcl_prob,
            labels,
            cls_loss_weights,
            gt_assignment,
            pc_labels,
            pc_probs,
            pc_count,
            img_cls_loss_weights,
            im_labels_real,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += ")"
        return tmpstr
