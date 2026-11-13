# Stable Diffusion Overview

Stable Diffusion Overview

Author

Benedict Thekkel

## Initial Checks

``` python
!conda list | grep "pytorch"
```

    pytorch                   2.0.1           py3.11_cuda11.8_cudnn8.7.0_0    pytorch
    pytorch-cuda              11.8                 h7e8668a_5    pytorch
    pytorch-ignite            0.4.12                   pypi_0    pypi
    pytorch-lightning         2.0.7                    pypi_0    pypi
    pytorch-mutex             1.0                        cuda    pytorch
    torchaudio                2.0.2               py311_cu118    pytorch
    torchtriton               2.0.0                     py311    pytorch
    torchvision               0.15.2              py311_cu118    pytorch

``` python
!pip list | grep "fastai" 
!pip list | grep "fastbook"
!pip list | grep "ipywidgets"
```

    fastai                        2.7.12
    fastbook                      0.0.28
    ipywidgets                    8.1.0

``` python
import torch
```

``` python
torch.cuda.is_available()
```

    True

## “UNET”

- input - somewhat noisy image
- output - the noise

image = image + noise

**Model trained to calculate noise in a image**

## Autoencoder = “VAE”

- output = input

**Model trained to compress and decompress images**

## Latents

latents = Autoencoders middle output (compressed version of the image)

## CLIP

CL - Contrast loss

**model trained to create image latents from text input**

Back to top
