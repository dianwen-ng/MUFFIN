import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import weight_norm, spectral_norm

from .meldataset import mel_spectrogram

from .modules.utils import get_padding, init_weights, GRN, Snake1d
from .modules.specFCQuantize import SpectraResidualVectorQuantize
from .modules.specEMAQuantize import emaSpectraResidualVectorQuantize


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        
        ## convolutions (with dilation)
        self.convs1 = nn.ModuleList([
            nn.Sequential(
                Snake1d(channels),
                weight_norm(
                        Conv1d(channels, channels // 2, kernel_size, 1,
                               dilation=d, padding=get_padding(kernel_size, d))
                )
            ) for d in dilation
        ])
        self.convs1.apply(init_weights)

        ## convolutions (without dilation)
        self.convs2 = nn.ModuleList([
            nn.Sequential(
                Snake1d(channels // 2), 
                weight_norm(
                    Conv1d(channels // 2, channels, kernel_size, 1,
                           dilation=1, padding=get_padding(kernel_size, 1))
                )
            ) for _ in dilation
        ])
        self.convs2.apply(init_weights)
        
        self.norm = GRN(channels)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(x)
            xt = c2(xt)
            x = xt + x
        x = self.norm(x)
        return x
    

class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            nn.Sequential(
                Snake1d(channels),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=d,
                           padding=get_padding(kernel_size, d))
                )
            ) for d in dilation
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = c(xt)
            x = xt + x
        return x
    
    
class Encoder(torch.nn.Module):
    def __init__(self, h):
        super(Encoder, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_downsamples = len(h.model_sample_rates)
        self.init_channel_dim = int(h.hidden_dims / (2**self.num_downsamples))
        
        ## pre-processing (convolution)
        self.conv_pre = weight_norm(Conv1d(1, self.init_channel_dim, 7, 1, padding=3))
        self.conv_pre.apply(init_weights)
        
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        
        ## build encoder (downsampling)
        self.downs = nn.ModuleList()
        self.resblock_post = nn.ModuleList()
        for i, (u, k) in enumerate(
            list(reversed(
                    list(zip(h.model_sample_rates, h.sample_kernel_sizes))
                ))
        ):
            self.downs.append(nn.Sequential(
                Snake1d(self.init_channel_dim * (2 ** i)), weight_norm(
                Conv1d(self.init_channel_dim * (2 ** i), self.init_channel_dim * (2 ** (i + 1)),
                       k, u, padding=(k - u) // 2))
                ))
            
            self.resblock_post.append(
                nn.Sequential(
                    GRN(self.init_channel_dim * (2 ** (i + 1))),
                    weight_norm(
                        Conv1d(self.init_channel_dim * (2 ** (i + 1)), 
                               self.init_channel_dim * (2 ** (i + 1)) * 3,
                               7, 1, 3, groups=h.conv_group)
                    ), Snake1d(self.init_channel_dim * (2 ** (i + 1)) * 3),
                    weight_norm(
                        Conv1d(self.init_channel_dim * (2 ** (i + 1)) * 3, 
                               self.init_channel_dim * (2 ** (i + 1)),
                               7, 1, 3, groups=h.conv_group)
                    )
                )
            )
                    
        self.downs.apply(init_weights)
        self.resblock_post.apply(init_weights)
        
        ## build encoder (residual)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            ch = self.init_channel_dim * (2 ** (i + 1))
            for j, (k, d) in enumerate( 
                zip(
                    h.resblock_kernel_sizes, h.down_resblock_dilation_sizes
                )
            ):
                self.resblocks.append(resblock(h, ch, k, d))
                
        ## build post-convolution processing
        self.conv_post = nn.Sequential(
            Snake1d(h.hidden_dims),
            weight_norm(Conv1d(h.hidden_dims, h.hidden_dims, 7, 1, padding=3)),
            
        )
        self.conv_post.apply(init_weights)

    def forward(self, x):
        ## pre-convolution processing
        x = self.conv_pre(x)

        ## feature encoding (compressing/downsampling)
        for i in range(self.num_downsamples):
            x = self.downs[i](x)
            
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
                    
            x = x + (xs / self.num_kernels)
            x = x + self.resblock_post[i](x)
        
        ## post-convolution processing
        x = self.conv_post(x)
        
        return x
        
        
class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.up_model_sample_rates)
        
        ## build pre-convolution for processing features
        self.conv_pre = weight_norm(
            Conv1d(h.hidden_dims, h.hidden_dims, 7, 1, padding=3)
        )
        self.conv_pre.apply(init_weights)
        
        ## build upsampler for compressed features
        self.ups = nn.ModuleList()
        self.resblock_post = nn.ModuleList()
        for i, (u, k) in enumerate(zip(
                h.up_model_sample_rates, h.up_sample_kernel_sizes
            )
        ):
            self.ups.append(
                nn.Sequential(
                    Snake1d(h.hidden_dims // (2**i)), weight_norm(
                    ConvTranspose1d(
                        h.hidden_dims // (2**i),
                        h.hidden_dims // (2**(i + 1)),
                        k, u, padding=(k - u) // 2,)
                ))
            )
            self.resblock_post.append(
                nn.Sequential(
                    GRN(h.hidden_dims // (2**(i + 1))),
                    weight_norm(
                        Conv1d(h.hidden_dims // (2**(i + 1)), 
                               h.hidden_dims // (2**(i + 1)) * 3,
                               7, 1, 3, groups=h.conv_group)
                    ), Snake1d(h.hidden_dims // (2**(i + 1)) * 3),
                    weight_norm(
                        Conv1d(h.hidden_dims // (2**(i + 1)) * 3, 
                               h.hidden_dims // (2**(i + 1)),
                               7, 1, 3, groups=h.conv_group)
                    )
                )
            )

        self.ups.apply(init_weights)
        self.resblock_post.apply(init_weights)

        ## build residual block for neural decoding
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.resblocks = nn.ModuleList()
        
        for i in range(len(self.ups)):
            ch = h.hidden_dims // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(
                h.resblock_kernel_sizes, h.up_resblock_dilation_sizes
            )):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = nn.Sequential(Snake1d(ch),
                                       weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
                                      )
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = self.ups[i](x)
                    
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = x + (xs / self.num_kernels)
            x = x + self.resblock_post[i](x)

        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    
class Quantizer(torch.nn.Module):
    def __init__(self, h):
        super(Quantizer, self).__init__()
        """ Codebook Residual Vector Quantizer
        Support codebook
        """
        self.h = h
        self.quantizer_type = self.h.quantizer_type
        
        if self.quantizer_type == 'spectral':
            ## perform quantization based on factor projection
            ## note: modules adopted from DAC taking reference from link below
            ## 'https://github.com/descriptinc/descript-audio-codec/tree/main'
            self.quantizer = SpectraResidualVectorQuantize(input_dim=h.hidden_dims, 
                                                           scales=h.scales,
                                                           codebook_size=h.n_codes,
                                                           code_dim=h.codebook_dims)
        
        elif self.quantizer_type == 'ema':
            ## perform quantization based on EMA
            ## note: modules adopted from Encodec taking reference from link below
            ## 'https://github.com/facebookresearch/encodec/tree/main/encodec'
            self.quantizer = emaSpectraResidualVectorQuantize(dimension = h.hidden_dims,
                                                              n_q = len(h.scales),
                                                              scales = h.scales,
                                                              bins = h.n_codes,
                                                              decay = 0.99,
                                                              kmeans_init = True,
                                                              kmeans_iters = 50,
                                                              threshold_ema_dead_code = 2,)
            self.frame_rate=h.frame_rate
            
        
        self.codebook_loss_lambda = self.h.codebook_loss_lambda
        self.commitment_loss_lambda = self.h.commitment_loss_lambda

    def forward(self, xin, return_codes=False):
        
        if self.quantizer_type == 'ema':
            qres = self.quantizer(xin, frame_rate=self.frame_rate)
            z_q, codes, loss = qres.quantized, qres.codes, qres.penalty
            
        elif self.quantizer_type == 'spectral':
            ## process high temporal resolution features
            z_q, codes, commitment_loss, codebook_loss = self.quantizer(xin)

            ## compute_loss
            commitment_loss = commitment_loss * self.commitment_loss_lambda
            codebook_loss = codebook_loss * self.codebook_loss_lambda

            loss = commitment_loss + codebook_loss
         
        return z_q, loss, codes
    

