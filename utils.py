import torch
import torch.nn.functional as F
from holowizard.core.reconstruction.constraints.window_2d import get_2d_window


def pad(image):
    size = [50 for x in image.shape for y in range(2)][-4:]

    mirrored_image = F.pad(
        image, size[::-1], mode="constant", value=0.0
    )
    mask = get_2d_window(
        mirrored_image.shape[-2:],
        [(50, mirrored_image.shape[-2] - 50), (50, mirrored_image.shape[-1] - 50)],
        [(120, 120), (120, 120)],
        "blackman",
        "cuda",
    )
    mirrored_image *= mask[None]
    return mirrored_image


def crop(input_array, crop_idcs):
    # Crop the array using the given slices
    cropped_array = input_array[
        ..., crop_idcs[0] : crop_idcs[1], crop_idcs[2] : crop_idcs[3]
    ]

    return cropped_array


def total_variation_loss(img, weight=1):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[..., 1:, :] - img[..., :-1, :], 2).sum()
    tv_w = torch.pow(img[..., 1:] - img[..., :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


def softmin_loss(d, tau=0.1, reduction="mean"):
    """
    Soft-min loss over distances.

    Args:
        d (Tensor): pairwise distances of shape [B, N]
                    (B = batch size, N = number of reference points).
        tau (float): temperature parameter (smaller tau -> sharper min).
        reduction (str): "mean", "sum", or "none" for output reduction.

    Returns:
        Tensor: scalar loss (if reduction != "none") or [B] loss values.
    """
    # -tau * logsumexp(-d/tau) along the reference dimension
    loss = -tau * torch.logsumexp(-d / tau, dim=0)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss  # shape [B]
