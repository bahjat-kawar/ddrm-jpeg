import torch
import torch.nn as nn
from .dct import LinearDCT, apply_linear_2d
import torch.nn.functional as F


def torch_rgb2ycbcr(x):
    # Assume x is a batch of size (N x C x H x W)
    v = torch.tensor([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]]).to(x.device)
    ycbcr = torch.tensordot(x, v, dims=([1], [1])).transpose(3, 2).transpose(2, 1)
    ycbcr[:,1:] += 128
    return ycbcr


def torch_ycbcr2rgb(x):
    # Assume x is a batch of size (N x C x H x W)
    v = torch.tensor([[ 1.00000000e+00, -3.68199903e-05,  1.40198758e+00],
       [ 1.00000000e+00, -3.44113281e-01, -7.14103821e-01],
       [ 1.00000000e+00,  1.77197812e+00, -1.34583413e-04]]).to(x.device)
    x[:, 1:] -= 128
    rgb = torch.tensordot(x, v, dims=([1], [1])).transpose(3, 2).transpose(2, 1)
    return rgb

def chroma_subsample(x):
    return x[:, 0:1, :, :], x[:, 1:, ::2, ::2]


def general_quant_matrix(qf = 10):
    q1 = torch.tensor([
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
    ]) 
    q2 = torch.tensor([
        17,  18,  24,  47,  99,  99,  99,  99,
        18,  21,  26,  66,  99,  99,  99,  99,
        24,  26,  56,  99,  99,  99,  99,  99,
        47,  66,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99
    ])
    s = (5000 / qf) if qf < 50 else (200 - 2 * qf)
    q1 = torch.floor((s * q1 + 50) / 100)
    q1[q1 <= 0] = 1
    q1[q1 > 255] = 255
    q2 = torch.floor((s * q2 + 50) / 100)
    q2[q2 <= 0] = 1
    q2[q2 > 255] = 255
    return q1, q2


def quantization_matrix(qf):
    return general_quant_matrix(qf)
    # q1 = torch.tensor([[ 80,  55,  50,  80, 120, 200, 255, 255],
    #                    [ 60,  60,  70,  95, 130, 255, 255, 255],
    #                    [ 70,  65,  80, 120, 200, 255, 255, 255],
    #                    [ 70,  85, 110, 145, 255, 255, 255, 255],
    #                    [ 90, 110, 185, 255, 255, 255, 255, 255],
    #                    [120, 175, 255, 255, 255, 255, 255, 255],
    #                    [245, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255]])
    # q2 = torch.tensor([[ 85,  90, 120, 235, 255, 255, 255, 255],
    #                    [ 90, 105, 130, 255, 255, 255, 255, 255],
    #                    [120, 130, 255, 255, 255, 255, 255, 255],
    #                    [235, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255]])
    # return q1, q2

def jpeg_encode(x, qf):
    # Assume x is a batch of size (N x C x H x W)
    # [-1, 1] to [0, 255]
    x = (x + 1) / 2 * 255
    n_batch, _, n_size, _ = x.shape
    
    x = torch_rgb2ycbcr(x)
    x_luma, x_chroma = chroma_subsample(x)
    unfold = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    x_luma = unfold(x_luma).transpose(2, 1)
    x_chroma = unfold(x_chroma).transpose(2, 1)

    x_luma = x_luma.reshape(-1, 8, 8) - 128
    x_chroma = x_chroma.reshape(-1, 8, 8) - 128
    
    dct_layer = LinearDCT(8, 'dct', norm='ortho')
    dct_layer.to(x_luma.device)
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)
    
    x_luma = x_luma.view(-1, 1, 8, 8)
    x_chroma = x_chroma.view(-1, 2, 8, 8)

    q1, q2 = quantization_matrix(qf)
    q1 = q1.to(x_luma.device)
    q2 = q2.to(x_luma.device)
    x_luma /= q1.view(1, 8, 8)
    x_chroma /= q2.view(1, 8, 8)
    
    x_luma = x_luma.round()
    x_chroma = x_chroma.round()
    
    x_luma = x_luma.reshape(n_batch, (n_size // 8) ** 2, 64).transpose(2, 1)
    x_chroma = x_chroma.reshape(n_batch, (n_size // 16) ** 2, 64 * 2).transpose(2, 1)
    
    fold = nn.Fold(output_size=(n_size, n_size), kernel_size=(8, 8), stride=(8, 8))
    x_luma = fold(x_luma)
    fold = nn.Fold(output_size=(n_size // 2, n_size // 2), kernel_size=(8, 8), stride=(8, 8))
    x_chroma = fold(x_chroma)
    
    return [x_luma, x_chroma]



def jpeg_decode(x, qf):
    # Assume x[0] is a batch of size (N x 1 x H x W) (luma)
    # Assume x[1:] is a batch of size (N x 2 x H/2 x W/2) (chroma)
    x_luma, x_chroma = x
    n_batch, _, n_size, _ = x_luma.shape
    unfold = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    x_luma = unfold(x_luma).transpose(2, 1)
    x_luma = x_luma.reshape(-1, 1, 8, 8)
    x_chroma = unfold(x_chroma).transpose(2, 1)
    x_chroma = x_chroma.reshape(-1, 2, 8, 8)
    
    q1, q2 = quantization_matrix(qf)
    q1 = q1.to(x_luma.device)
    q2 = q2.to(x_luma.device)
    x_luma *= q1.view(1, 8, 8)
    x_chroma *= q2.view(1, 8, 8)
    
    x_luma = x_luma.reshape(-1, 8, 8)
    x_chroma = x_chroma.reshape(-1, 8, 8)
    
    dct_layer = LinearDCT(8, 'idct', norm='ortho')
    dct_layer.to(x_luma.device)
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)
    
    x_luma = (x_luma + 128).reshape(n_batch, (n_size // 8) ** 2, 64).transpose(2, 1)
    x_chroma = (x_chroma + 128).reshape(n_batch, (n_size // 16) ** 2, 64 * 2).transpose(2, 1)

    fold = nn.Fold(output_size=(n_size, n_size), kernel_size=(8, 8), stride=(8, 8))
    x_luma = fold(x_luma)
    fold = nn.Fold(output_size=(n_size // 2, n_size // 2), kernel_size=(8, 8), stride=(8, 8))
    x_chroma = fold(x_chroma)

    x_chroma_repeated = torch.zeros(n_batch, 2, n_size, n_size, device = x_luma.device)
    x_chroma_repeated[:, :, 0::2, 0::2] = x_chroma
    x_chroma_repeated[:, :, 0::2, 1::2] = x_chroma
    x_chroma_repeated[:, :, 1::2, 0::2] = x_chroma
    x_chroma_repeated[:, :, 1::2, 1::2] = x_chroma

    x = torch.cat([x_luma, x_chroma_repeated], dim=1)

    x = torch_ycbcr2rgb(x)
    
    # [0, 255] to [-1, 1]
    x = x / 255 * 2 - 1
    
    return x

def quantization_encode(x, qf):
    qf = 32
    #to int
    x = (x + 1) / 2
    x = x * 255
    x = x.int()
    # quantize
    x = x // qf
    #to float
    x = x.float()
    x = x / (255/qf)
    x = (x * 2) - 1
    return x

def quantization_decode(x, qf):
    return x