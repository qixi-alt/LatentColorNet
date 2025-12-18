import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
import matplotlib.pyplot as plt
import os


def codebook_usage_loss(indices, n_embed, eps=1e-5):
    """
    Compute the negative entropy of code usage to encourage more uniform activation of codewords.

    Args:
        indices: Tensor of shape [B, H, W] or [B*H*W], the quantized output code indices
        n_embed: int, size of the codebook (number of codewords)
        eps: float, a small constant to avoid division by zero

    Returns:
        usage_loss: scalar. A higher value indicates more uneven usage. When added to the main loss, it acts as a positive penalty term.
    """

    flat_indices = indices.view(-1)

    counts = torch.bincount(flat_indices, minlength=n_embed).float()

    probs = (counts + eps) / (counts.sum() + eps * n_embed)

    entropy = -torch.sum(probs * torch.log(probs + eps))

    max_entropy = torch.log(torch.tensor(n_embed, dtype=torch.float32))
    usage_loss = max_entropy - entropy

    return usage_loss


class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class EMAQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_e = n_e  # codebook size
        self.e_dim = e_dim  # embedding dimension
        self.beta = beta
        self.decay = decay
        self.eps = eps

        embed = torch.randn(n_e, e_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(n_e))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z):
        # z: [B, C, H, W]
        B, C, H, W = z.shape
        z = rearrange(z, 'b c h w -> b h w c')  # [B, H, W, C]
        z_flat = z.reshape(-1, self.e_dim)  # [BHW, C]

        # compute distance
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding ** 2, dim=1) - 2 * torch.matmul(z_flat, self.embedding.t())  # [BHW, n_e]

        indices = torch.argmin(d, dim=1)  # [BHW]
        z_q = self.embedding[indices]  # [BHW, C]
        z_q = z_q.view(B, H, W, C)
        z_q = rearrange(z_q, 'b h w c -> b c h w')  # 回到 [B, C, H, W]

        # EMA update
        if self.training:
            one_hot = F.one_hot(indices, self.n_e).type_as(z_flat)  # [BHW, n_e]
            cluster_size = one_hot.sum(0)

            embed_sum = torch.matmul(one_hot.t(), z_flat)

            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) / (n + self.n_e * self.eps)) * n

            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)

        # compute commitment loss
        loss = self.beta * F.mse_loss(z_q.detach(), z.permute(0, 3, 1, 2))  # 保证维度一致

        # straight-through
        z = rearrange(z, 'b h w c -> b c h w')
        z_q = z + (z_q - z).detach()

        # reshape indices
        indices = indices.view(B, H, W)

        return z_q, loss, (None, None, indices)

    def get_codebook_entry(self, indices, shape):
        # indices: [B, H, W] or [B, H*W]
        B = indices.shape[0]
        if len(indices.shape) == 2:
            H = W = int((indices.shape[1]) ** 0.5)
            indices = indices.view(B, H, W)

        z_q = self.embedding[indices.view(-1)]  # [B*H*W, C]
        z_q = z_q.view(B, H, W, self.e_dim)
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_codebooks, n_embed, embed_dim, patch_size=4,beta=0.25, sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.sane_index_shape = sane_index_shape

        # 创建多个码本 (每个阶段一个量化器)
        # self.quantizers = nn.ModuleList([
        #     VectorQuantizer2(
        #         n_embed,
        #         embed_dim,
        #         beta,
        #         remap=None,  # 残差量化不支持索引重映射
        #         unknown_index="random",
        #         sane_index_shape=False,  # 内部统一使用展平索引
        #         legacy=legacy
        #     ) for _ in range(n_codebooks)
        #

        self.quantizers = nn.ModuleList([
            EMAQuantizer(
                n_embed,
                embed_dim,
                beta,
                decay=0.99
            ) for _ in range(n_codebooks)
        ])

    def forward(self, z):

        # z_q = torch.zeros_like(z)
        # residual = z
        # losses = []
        # all_indices = []
        #
        # residual_errors = []

        B, C, H, W = z.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"输入大小必须能整除 patch_size={self.patch_size}"

        z_patches = rearrange(z, 'b c (h ph) (w pw) -> (b h w) c ph pw',
                              ph=self.patch_size, pw=self.patch_size)
        N = z_patches.shape[0]

        z_q_patches = torch.zeros_like(z_patches)
        residual = z_patches
        losses = []
        all_indices = []
        residual_errors = []

        for quantizer in self.quantizers:
            # 量化当前残差
            # z_q_i, loss_i, (_, _, indices_i) = quantizer(residual, temp, rescale_logits, return_logits)
            z_q_i, loss_i, (_, _, indices_i) = quantizer(residual)

            usage_loss_i = codebook_usage_loss(indices_i, quantizer.n_e)
            loss_i = loss_i + 0.01 * usage_loss_i  # 0.01 是权重，可以调大点试试

            # 这里保留数值，保留残差梯度！
            # z_q = z_q + z_q_i
            z_q_patches = z_q_patches + z_q_i

            residual = residual - z_q_i

            error = torch.mean(residual ** 2).item()
            residual_errors.append(error)

            residual = residual.detach() + (residual - residual.detach())

            losses.append(loss_i)
            # all_indices.append(indices_i.squeeze(-1))  # [B, H*W]
            indices_flat = indices_i.view(indices_i.shape[0], -1)  # 展平成 patch_area
            all_indices.append(indices_flat)
            print(f"Residual MSE after layer {len(all_indices)}: {error:.6f}")

        # loss = torch.stack(losses).sum()
        # min_encoding_indices = torch.stack(all_indices, dim=-1)
        # if self.sane_index_shape:
        #     b, c, h, w = z.shape
        #     min_encoding_indices = min_encoding_indices.reshape(b, h, w, self.n_codebooks)
        #
        # return z_q, loss, (None, None, min_encoding_indices)

        loss = torch.stack(losses).sum()

        z_q = rearrange(z_q_patches, '(b h w) c ph pw -> b c (h ph) (w pw)',
                        h=H // self.patch_size, w=W // self.patch_size,
                        ph=self.patch_size, pw=self.patch_size)

        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        patch_area = self.patch_size * self.patch_size
        min_encoding_indices = torch.stack(all_indices, dim=-1)  # [B*N, patch_area, L]
        min_encoding_indices = rearrange(min_encoding_indices,
                                         '(b h w) a l -> b h w a l',
                                         h=patch_h, w=patch_w)

        return z_q, loss, (None, None, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        """从多码本索引重建特征"""
        if self.sane_index_shape:
            b, h, w, n_c = indices.shape
            indices = indices.reshape(b, h * w, n_c)
        else:
            b, len, n_c = indices.shape

        z_q = torch.zeros(b, self.quantizers[0].e_dim, *shape[2:],
                          device=indices.device)

        # 遍历所有码本并累加重建结果
        for i, quantizer in enumerate(self.quantizers):
            indices_i = indices[..., i].contiguous()
            z_q += quantizer.get_codebook_entry(indices_i, shape)

        return z_q

    def visualize_codebook_usage(self, all_indices, save_path="codebook_usage.png"):
        """
        可视化每个码本的使用频率直方图，并保存图像。

        参数:
            all_indices: Tensor [B, H*W, n_codebooks]
            save_path: 保存路径，例如 "logs/codebook_usage.png"
        """

        n_codebooks = all_indices.shape[-1]
        n_embed = self.quantizers[0].n_e  # 每个量化器的码字数量

        usage_counts = []
        for i in range(n_codebooks):
            indices_i = all_indices[..., i].reshape(-1)
            counts = torch.bincount(indices_i, minlength=n_embed).cpu().numpy()
            usage_counts.append(counts)

        fig, axs = plt.subplots(n_codebooks, 1, figsize=(12, 2.5 * n_codebooks), sharex=True)
        for i, counts in enumerate(usage_counts):
            axs[i].bar(range(n_embed), counts, color='gray')
            axs[i].set_title(f"Codebook {i + 1} Usage Histogram")
            axs[i].set_ylabel("Frequency")
        axs[-1].set_xlabel("Code Index")
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

