import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


def dense_crf(
    imgs,
    probs,
    max_iter=10,
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
):

    # DeepLabv2_VGG16:
    # max_iter=10
    # size_std=513
    # pos_w=2
    # pos_x_std=2
    # pos_y_std=2
    # bi_w=4
    # bi_x_std=65
    # bi_y_std=65
    # bi_r_std=3
    # bi_g_std=3
    # bi_b_std=3

    # DeepLabv2_ResNet101:
    # max_iter=10
    # size_std=513
    # pos_w=3
    # pos_x_std=1
    # pos_y_std=1
    # bi_w=4
    # bi_x_std=67
    # bi_y_std=67
    # bi_r_std=3
    # bi_g_std=3
    # bi_b_std=3

    N, C, H, W = probs.shape

    scale_factor = 1.0 * size_std / max(H, W)
    # scale_factor = 1.0

    probs = np.ascontiguousarray(probs, np.float32)
    imgs = np.ascontiguousarray(imgs, dtype=np.uint8)

    Qs = np.zeros_like(probs)

    for n in range(N):
        prob = probs[n]
        img = imgs[n]

        d = dcrf.DenseCRF2D(W, H, C)

        U = unary_from_softmax(prob)
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(
            sxy=(pos_x_std / scale_factor, pos_y_std / scale_factor), compat=pos_w
        )

        d.addPairwiseBilateral(
            sxy=(bi_x_std / scale_factor, bi_y_std / scale_factor),
            srgb=(bi_r_std, bi_g_std, bi_b_std),
            rgbim=img,
            compat=bi_w,
        )

        Q = d.inference(max_iter)
        # MAP = np.argmax(Q, axis=0).reshape((H, W))
        Qs[n, ...] = np.array(Q).reshape(C, H, W)

    return Qs
