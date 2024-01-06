# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/150_datasets.ipynb.

# %% ../nbs/150_datasets.ipynb 3
from __future__ import annotations
import math,numpy as np,matplotlib.pyplot as plt
from operator import itemgetter
from itertools import zip_longest
import fastcore.all as fc

from torch.utils.data import default_collate

from .training import *

# %% auto 0
__all__ = ['collate_fn', 'transforms', 'inplace']

# %% ../nbs/150_datasets.ipynb 4
import logging,pickle,gzip,os,time,shutil,torch,matplotlib as mpl
from pathlib import Path

from torch import tensor,nn,optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import load_dataset,load_dataset_builder

import torchvision.transforms.functional as TF
from fastcore.test import test_close

# %% ../nbs/150_datasets.ipynb 19
def collate_fn(b):
    return {x:torch.stack([TF.to_tensor(o[x]) for o in b]),
            y:tensor([o[y] for o in b])}

# %% ../nbs/150_datasets.ipynb 21
def transforms(b):
    b[x] = [TF.to_tensor(o) for o in b[x]]
    return b

# %% ../nbs/150_datasets.ipynb 23
def _transformi(b): b[x] = [torch.flatten(TF.to_tensor(o)) for o in b[x]]

# %% ../nbs/150_datasets.ipynb 24
def inplace(f):
    def _f(b):
        f(b)
        return b
    return _f

# %% ../nbs/150_datasets.ipynb 43
from nbdev.showdoc import show_doc
