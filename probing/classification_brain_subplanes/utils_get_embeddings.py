import os
import json

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as T
import albumentations as A
import open_clip
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

from tqdm import tqdm
from PIL import Image

from utils import (
    DatasetPlanesDB,
    DIR_TEST,
    LIST_TEST_PID,
)

def get_list_embeddings(model, dl):
    list_z = []

    for x in tqdm(dl):
        x = x[0].to('cuda')
        with torch.no_grad():
            z = model(x)
        list_z.append(z.detach().cpu())

    z = torch.cat(list_z, dim=0)

    return [t.numpy() for t in z]

def get_dict_embeddings(
        model, ds,
        batch_size=8, num_workers=4,
    ):
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    list_embeddings = get_list_embeddings(model, dl)

    dict_embeddings = {}
    for filepath, embedding in tqdm(zip(ds.data, list_embeddings)):
        filename = os.path.basename(filepath).split('.')[0]
        dict_embeddings[filename] = embedding
    
    return dict_embeddings

def get_test_embeddings(
        model, preprocess_img,
        batch_size=8, num_workers=4,
    ):
    ds = DatasetPlanesDB(
        DIR_TEST,
        LIST_TEST_PID,
        preprocess_img,
    )

    dict_embeddings = get_dict_embeddings(model, ds, batch_size, num_workers)

    return dict_embeddings

def get_train_val_embeddings(
        model, root_train, root_val, train_val_pid,
        preprocess_img,
        batch_size=8, num_workers=4,
    ):
    ds_train = DatasetPlanesDB(
        root_train,
        train_val_pid[0],
        preprocess_img,
    )
    ds_val = DatasetPlanesDB(
        root_val,
        train_val_pid[1],
        preprocess_img,
    )

    dict_train_embeddings = get_dict_embeddings(model, ds_train, batch_size, num_workers)
    dict_val_embeddings   = get_dict_embeddings(model, ds_val,   batch_size, num_workers)

    return dict_train_embeddings, dict_val_embeddings