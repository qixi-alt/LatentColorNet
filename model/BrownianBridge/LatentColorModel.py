import itertools
import pdb
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from model.BrownianBridge.BaseModel import BaseModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQModel

import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class GaborFeatureExtractor(nn.Module):
    def __init__(self, ksize=31, lambd=3, gamma=0.5, orientations=[0, np.pi / 2]):
        super().__init__()
        self.ksize = ksize
        self.lambd = lambd
        self.gamma = gamma
        self.orientations = orientations
        self.padding = ksize // 2

        self.register_buffers()

    def register_buffers(self):
        kernels_real = []
        kernels_imag = []
        sigma = self.lambd / 2

        for theta in self.orientations:
            kernel_real = cv2.getGaborKernel(
                (self.ksize, self.ksize),
                sigma,
                theta,
                self.lambd,
                self.gamma,
                0,
                ktype=cv2.CV_32F
            )
            kernel_imag = cv2.getGaborKernel(
                (self.ksize, self.ksize),
                sigma,
                theta,
                self.lambd,
                self.gamma,
                np.pi / 2,
                ktype=cv2.CV_32F
            )
            kernels_real.append(torch.FloatTensor(kernel_real))
            kernels_imag.append(torch.FloatTensor(kernel_imag))

        self.register_buffer('kernels_real', torch.stack(kernels_real))
        self.register_buffer('kernels_imag', torch.stack(kernels_imag))

    def to_grayscale(self, x):
        if x.shape[1] == 3:  # RGB图像
            return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        return x[:, 0]  # 已经是单通道

    def forward(self, x):
        x_gray = self.to_grayscale(x).unsqueeze(1)  # [B, 1, H, W]
        real = F.conv2d(
            x_gray,
            self.kernels_real.unsqueeze(1),  # [num_orient, 1, ksize, ksize]
            padding=self.padding
        )
        imag = F.conv2d(
            x_gray,
            self.kernels_imag.unsqueeze(1),  # [num_orient, 1, ksize, ksize]
            padding=self.padding
        )
        magnitude = torch.sqrt(real ** 2 + imag ** 2)  # [B, num_orient, H, W]
        gabor_feature, _ = torch.max(magnitude, dim=1, keepdim=True)

        min_val = gabor_feature.reshape(gabor_feature.shape[0], -1).min(dim=1)[0]
        max_val = gabor_feature.reshape(gabor_feature.shape[0], -1).max(dim=1)[0]
        min_val = min_val.view(-1, 1, 1, 1)
        max_val = max_val.view(-1, 1, 1, 1)
        gabor_feature = (gabor_feature - min_val) / (max_val - min_val + 1e-8)

        return gabor_feature


class LatentColorModel(BaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.gabor_extractor = GaborFeatureExtractor(
            ksize=31,
            lambd=3,  # 小尺度特征
            gamma=0.5,
            orientations=[0, np.pi / 2]  # 水平和垂直方向
        )

        self.gate_conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, x_cond, context=None):
        with torch.no_grad():

            gabor_prior = self.gabor_extractor(x_cond)
            combined = torch.cat([x_cond, gabor_prior], dim=1)
            channel_weights = torch.sigmoid(self.gate_conv(combined))

            x_cond_enhanced = channel_weights * x_cond + (1 - channel_weights) * gabor_prior

            x_cond_latent = self.encode(x_cond_enhanced, cond=True)
            x_latent = self.encode(x, cond=False)

        context = self.get_cond_stage_context(x_cond)
        return super().forward(x_latent.detach(), x_cond_latent.detach(), context)


    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)

        x_latent_quant, loss, (_, _, indices) = model.quantize(x_latent)
        model.quantize.last_indices = indices
        out = model.decode(x_latent_quant)
        return out

    @torch.no_grad()
    def sample(self, x_cond, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode(x_cond, cond=True)
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=self.get_cond_stage_context(x_cond),
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop(y=x_cond_latent,
                                      context=self.get_cond_stage_context(x_cond),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            out = self.decode(x_latent, cond=False)
            return out

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

    # @torch.no_grad()
    # def reverse_sample(self, x, skip=False):
    #     x_ori_latent = self.vqgan.encoder(x)
    #     temp, _ = self.brownianbridge.reverse_p_sample_loop(x_ori_latent, x, skip=skip, clip_denoised=False)
    #     x_latent = temp[-1]
    #     x_latent = self.vqgan.quant_conv(x_latent)
    #     x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
    #     out = self.vqgan.decode(x_latent_quant)
    #     return out
