"""Training-only geometry, camera-style noise, and a small border crop.

The homography matches utils.get_rand_transform_matrix and
tf.contrib.image.transform: the 3×3 matrix maps an output pixel to the input
pixel it should sample. Encoding at test time does not use any of this.
"""

import math
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def ramp(step, length):
    if length <= 0:
        return 1.0
    return float(min(step / float(length), 1.0))


def random_homographies(batch, image_size, max_corner_shift):
    """Return (forward, inverse) matrices, each [B, 3, 3], numpy float32.

    forward is what tf.contrib.image.transform receives as M[:, 1].
    inverse is M[:, 0].
    """
    forwards, inverses = [], []
    dst = np.array(
        [[0, 0], [image_size, 0], [image_size, image_size], [0, image_size]],
        dtype=np.float32,
    )
    d = float(max_corner_shift)
    for _ in range(batch):
        jitter = [(random.uniform(-d, d), random.uniform(-d, d)) for _ in range(4)]
        tl, tr, br, bl = jitter
        rect = np.array(
            [
                [tl[0], tl[1]],
                [tr[0] + image_size, tr[1]],
                [br[0] + image_size, br[1] + image_size],
                [bl[0], bl[1] + image_size],
            ],
            dtype=np.float32,
        )
        try:
            matrix = cv2.getPerspectiveTransform(rect, dst)
            inverse = np.linalg.inv(matrix)
            if not (np.isfinite(matrix).all() and np.isfinite(inverse).all()):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            matrix = np.eye(3, dtype=np.float32)
            inverse = np.eye(3, dtype=np.float32)
        forwards.append(matrix.astype(np.float32))
        inverses.append(inverse.astype(np.float32))
    return np.stack(forwards), np.stack(inverses)


def perspective_warp(image, matrix):
    """Sample `image` with an output→input homography. `matrix` is [B, 3, 3]."""
    batch, _, height, width = image.shape
    homography = matrix
    if homography.dim() == 2:
        homography = homography.unsqueeze(0)
    if homography.shape[-2:] != (3, 3):
        raise ValueError("homography must be [B, 3, 3], got %s" % (tuple(homography.shape),))
    if homography.shape[0] not in (1, batch):
        raise ValueError(
            "homography batch %d != image batch %d (image %s). "
            "Use random_homographies(cover.shape[0], image_size, shift), not (400, batch, ...)."
            % (homography.shape[0], batch, tuple(image.shape))
        )
    if homography.shape[0] == 1:
        homography = homography.expand(batch, -1, -1)
    homography = homography.to(device=image.device, dtype=image.dtype)

    ys, xs = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=image.dtype),
        torch.arange(width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    coords = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(height * width, 3)
    pixels = coords.unsqueeze(0).expand(batch, -1, -1)
    transform = homography.transpose(-1, -2).contiguous()
    source = torch.bmm(pixels, transform).reshape(batch, height, width, 3)
    source = source[..., :2] / source[..., 2:3].clamp(min=1e-8)
    grid_x = source[..., 0] / (width - 1) * 2 - 1
    grid_y = source[..., 1] / (height - 1) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(
        image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


def warp_cover(cover, forward):
    """Warp the cover and fill samples that landed outside it with the original pixels."""
    warped = perspective_warp(cover, forward)
    mask = perspective_warp(torch.ones_like(cover), forward)
    return warped + (1.0 - mask) * cover, mask


def unwarp_white(image, inverse):
    """Warp back to the square and fill the empty border with white."""
    warped = perspective_warp(image, inverse)
    mask = perspective_warp(torch.ones_like(image), inverse)
    return warped + (1.0 - mask)


def _blur_kernel(device):
    size = 7
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(size, device=device, dtype=torch.float32),
            torch.arange(size, device=device, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    )
    coords = coords - 0.5 * (size - 1)
    manhattan = coords.abs().sum(-1)
    pick = torch.rand((), device=device)
    if float(pick) < 0.25:
        sigma = torch.empty((), device=device).uniform_(1.0, 3.0)
        values = torch.exp(-(coords ** 2).sum(-1) / (2 * sigma ** 2))
    elif float(pick) < 0.5:
        theta = torch.empty((), device=device).uniform_(0, 2 * math.pi)
        direction = torch.stack([torch.cos(theta), torch.sin(theta)])
        distance = (coords * direction).sum(-1)
        sigma = torch.empty((), device=device).uniform_(0.25, 1.0)
        width = torch.empty((), device=device).uniform_(3.0, 0.5 * (size - 1) + 0.1)
        values = torch.exp(-distance ** 2 / (2 * sigma ** 2)) * (manhattan < width).float()
    else:
        values = (manhattan < 0.5).float()
    values = values / values.sum().clamp(min=1e-8)
    return values.view(1, 1, size, size).repeat(3, 1, 1, 1)


def photometric(image, step, cfg):
    """Blur, noise, contrast, brightness, hue, saturation, then differentiable JPEG.

    Strengths follow train.py defaults and scripts/base.sh. `cfg` is TrainConfig.
    """
    device = image.device
    batch = image.shape[0]

    blur_level = ramp(step, cfg.noise_ramp)
    if blur_level > 0:
        blurred = F.conv2d(image, _blur_kernel(device), padding=3, groups=3)
        blurred = image * (1.0 - blur_level) + blurred * blur_level
    else:
        blurred = image

    noise_strength = torch.rand((), device=device) * ramp(step, cfg.noise_ramp) * cfg.noise
    noised = torch.clamp(blurred + torch.randn_like(blurred) * noise_strength, 0, 1)

    contrast_level = ramp(step, cfg.contrast_ramp)
    low = 1.0 - (1.0 - cfg.contrast_low) * contrast_level
    high = 1.0 + (cfg.contrast_high - 1.0) * contrast_level
    contrast = torch.empty(batch, 1, 1, 1, device=device).uniform_(low, high)
    bri_amp = ramp(step, cfg.bri_ramp) * cfg.bri
    hue_amp = ramp(step, cfg.hue_ramp) * cfg.hue
    brightness = torch.empty(batch, 1, 1, 1, device=device).uniform_(-bri_amp, bri_amp)
    hue = torch.empty(batch, 3, 1, 1, device=device).uniform_(-hue_amp, hue_amp)
    colored = torch.clamp(noised * contrast + brightness + hue, 0, 1)

    sat = torch.rand((), device=device) * ramp(step, cfg.sat_ramp) * cfg.sat
    luma_w = torch.tensor([0.3, 0.6, 0.1], device=device).view(1, 3, 1, 1)
    luma = (colored * luma_w).sum(dim=1, keepdim=True)
    colored = (1.0 - sat) * colored + sat * luma

    quality_floor = 100.0 - (100.0 - cfg.jpeg_quality) * ramp(step, cfg.jpeg_ramp)
    quality = 100.0 - random.random() * (100.0 - quality_floor)
    return jpeg_compress(colored, quality_to_factor(quality))


def quality_to_factor(quality):
    quality = max(float(quality), 1.0)
    if quality < 50:
        raw = 5000.0 / quality
    else:
        raw = 200.0 - quality * 2.0
    return raw / 100.0 + 1e-4


def random_border_crop(image, max_fraction):
    """Trim the same fraction off every edge and resize back to 400×400.

    This is the objective-3 augmentation: a small crop that still contains the
    center. `max_fraction` of 0 leaves the image unchanged.
    """
    if max_fraction <= 0:
        return image
    batch, _, height, width = image.shape
    outputs = []
    for index in range(batch):
        fraction = random.random() * max_fraction
        margin = int(fraction * height)
        if margin < 1 or margin * 2 >= height:
            outputs.append(image[index])
            continue
        cropped = image[index : index + 1, :, margin : height - margin, margin : width - margin]
        outputs.append(F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)[0])
    return torch.stack(outputs, dim=0)


# --- Differentiable JPEG, ported from utils.jpeg_compress_decompress ---

def _dct_tensors():
    forward = np.zeros((8, 8, 8, 8), np.float32)
    inverse = np.zeros((8, 8, 8, 8), np.float32)
    for x in range(8):
        for y in range(8):
            for u in range(8):
                for v in range(8):
                    forward[x, y, u, v] = math.cos((2 * x + 1) * u * math.pi / 16) * math.cos(
                        (2 * y + 1) * v * math.pi / 16
                    )
                    inverse[x, y, u, v] = math.cos((2 * u + 1) * x * math.pi / 16) * math.cos(
                        (2 * v + 1) * y * math.pi / 16
                    )
    alpha = np.array([1.0 / math.sqrt(2.0)] + [1.0] * 7, np.float32)
    scale = np.outer(alpha, alpha) * 0.25
    alpha_outer = np.outer(alpha, alpha)
    return forward, scale, inverse, alpha_outer


_FWD, _SCALE, _INV, _ALPHA = _dct_tensors()

_Y_TABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
).T

_C_TABLE = np.full((8, 8), 99, dtype=np.float32)
_C_TABLE[:4, :4] = np.array(
    [[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]],
    dtype=np.float32,
).T

_RGB_TO_YCBCR = np.array(
    [
        [0.299, -0.168736, 0.5],
        [0.587, -0.331264, -0.418688],
        [0.114, 0.5, -0.081312],
    ],
    dtype=np.float32,
)
_YCBCR_TO_RGB = np.array(
    [[1.0, 1.0, 1.0], [0.0, -0.344136, 1.772], [1.402, -0.714136, 0.0]],
    dtype=np.float32,
)


def _as_tensor(array, device, dtype):
    return torch.as_tensor(array, device=device, dtype=dtype)


def _round_only_at_0(values):
    near = (values.abs() < 0.5).float()
    return near * (values ** 3) + (1.0 - near) * values


def _to_patches(component):
    batch, height, width = component.shape
    block = 8
    component = component.view(batch, height // block, block, width // block, block)
    component = component.permute(0, 1, 3, 2, 4)
    return component.reshape(batch, -1, block, block)


def _from_patches(patches, height, width):
    batch = patches.shape[0]
    block = 8
    image = patches.view(batch, height // block, width // block, block, block)
    image = image.permute(0, 1, 3, 2, 4)
    return image.reshape(batch, height, width)


def jpeg_compress(image, factor):
    """RGB [B,3,H,W] in [0, 1] → approximated JPEG round-trip, still differentiable."""
    device, dtype = image.device, image.dtype
    scaled = image * 255.0
    red = scaled[:, 0, :, :]
    green = scaled[:, 1, :, :]
    blue = scaled[:, 2, :, :]
    matrix = _as_tensor(_RGB_TO_YCBCR, device, dtype)
    channels = torch.stack([red, green, blue], dim=-1)
    ycbcr = torch.matmul(channels, matrix) + torch.tensor([0.0, 128.0, 128.0], device=device, dtype=dtype)
    y = ycbcr[..., 0]
    cb = ycbcr[..., 1]
    cr = ycbcr[..., 2]
    cb = F.avg_pool2d(cb.unsqueeze(1), 2).squeeze(1)
    cr = F.avg_pool2d(cr.unsqueeze(1), 2).squeeze(1)

    fwd = _as_tensor(_FWD, device, dtype)
    scale = _as_tensor(_SCALE, device, dtype)
    inv = _as_tensor(_INV, device, dtype)
    alpha = _as_tensor(_ALPHA, device, dtype)
    y_table = _as_tensor(_Y_TABLE, device, dtype)
    c_table = _as_tensor(_C_TABLE, device, dtype)
    factor = torch.as_tensor(factor, device=device, dtype=dtype)

    def _block_roundtrip(component, table, height, width):
        patches = _to_patches(component) - 128.0
        coeffs = torch.einsum("bnxy,xyuv->bnuv", patches, fwd) * scale
        coeffs = _round_only_at_0(coeffs / (table * factor))
        coeffs = coeffs * (table * factor)
        restored = 0.25 * torch.einsum("bnxy,xyuv->bnuv", coeffs * alpha, inv) + 128.0
        return _from_patches(restored, height, width)

    height, width = y.shape[-2:]
    y = _block_roundtrip(y, y_table, height, width)
    cb = _block_roundtrip(cb, c_table, height // 2, width // 2)
    cr = _block_roundtrip(cr, c_table, height // 2, width // 2)
    cb = cb.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    cr = cr.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

    stacked = torch.stack([y, cb, cr], dim=-1) + torch.tensor([0.0, -128.0, -128.0], device=device, dtype=dtype)
    rgb = torch.matmul(stacked, _as_tensor(_YCBCR_TO_RGB, device, dtype))
    rgb = rgb.permute(0, 3, 1, 2)
    return torch.clamp(rgb, 0.0, 255.0) / 255.0
