from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from .utils import Snake1d, get_padding, LayerNorm


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        Factorized codes: Perform nearest neighbor lookup in low-dimensional space
        for improved codebook usage
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        
        if input_dim != codebook_dim:
            self.in_proj = weight_norm(nn.Conv1d(input_dim, codebook_dim, kernel_size=1))
            self.out_proj = weight_norm(nn.Conv1d(codebook_dim, input_dim, kernel_size=1))
        
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        
        ## define codebook tokenizer (random initialize)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        """

        ## Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        
        z_q, indices = self.decode_latents(z_e)
        
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
     
        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        ## L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        encodings = F.dropout(encodings, p=0.15 if self.training else 0)
        
        codebook = F.normalize(codebook)
        codebook = F.dropout(codebook, p=0.15 if self.training else 0)

        ## Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class SpectraResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """
    def __init__(
        self,
        input_dim: int = 512,
        scales: List[int] = [4, 2, 1, 1],
        codebook_size: int = 1024,
        code_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.scales = scales
        self.n_codebooks = len(self.scales)
        self.codebook_size = codebook_size
        self.codebook_dim = code_dim

        self.quantizers = nn.ModuleList(
            [VectorQuantize(input_dim, codebook_size, code_dim) for _ in self.scales]
        )

    def forward(self, z):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []

        for i, quantizer in enumerate(self.quantizers):

            z_q_i, commitment_loss_i, codebook_loss_i, codes_i = quantizer(
                self.fft_bandfilter(residual, scale_factor=self.scales[i])
            )
            
            z_q = z_q + z_q_i
            codebook_loss_i, residual = self.bandfiltered_codebook_loss(residual, z_q_i, 
                                                                        codebook_loss_i,
                                                                        self.scales[i])
            
            ## sum losses
            commitment_loss += (commitment_loss_i).mean()
            codebook_loss += (codebook_loss_i).mean()

            codebook_indices.append(codes_i)

        codes = torch.stack(codebook_indices, dim=1)

        return z_q, codes, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor, use_bookindex: List[int]=[0, 1, 2, 3]):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_remove = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)
            z_q_i = self.quantizers[i].out_proj(z_p_i)
            
            ## for model analysis (record specific codebooks for removal)
            if i not in use_bookindex:
                z_remove += z_q_i
                
            z_q = z_q + z_q_i
        
        ## remove specific features of codebook for codebook analysis
        z_q = z_q - z_remove
        return z_q, torch.cat(z_p, dim=1), codes
    
    def fft_bandfilter(self, x, scale_factor=4):
        
        original_length = x.size(-1)
        
        if scale_factor != 1:
            x = torch.fft.rfft(x, dim=-1)
            filter_length = int(x.size(-1) // scale_factor)
            x[..., filter_length + 1:] = 0

            x = torch.fft.irfft(x, n=original_length, dim=-1)

        return x
    
    ## codebook for residuals
    def bandfiltered_codebook_loss(self, in_feat, codebook_feats, 
                                   codebook_loss, scale_factor, alpha=0.9):
        
        residual = in_feat - codebook_feats
        
        if scale_factor > 1 and self.training:
            in_feat = in_feat.detach()
            tar_residuals = in_feat - self.fft_bandfilter(in_feat, scale_factor)
            residual_loss = F.mse_loss(residual, tar_residuals, reduction="none").mean([1, 2])

            codebook_loss = (alpha * codebook_loss) + ((1-alpha) * residual_loss)

        return codebook_loss, residual
    


if __name__ == "__main__":
    rvq = ResidualVectorQuantizeBS()
    x = torch.randn(16, 512, 80)
    y = rvq(x)
