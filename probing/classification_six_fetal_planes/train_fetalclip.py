import os

import json
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import torchvision.transforms as T
import albumentations as A
import open_clip

from torch.utils.data import DataLoader

from utils import *
from utils_get_embeddings import (
    get_test_embeddings,
    get_train_val_embeddings,
)

with open('config.json', 'r') as file:
    config = json.load(file)

PATH_FETALCLIP_WEIGHT = config['paths']['path_fetalclip_weight']
PATH_FETALCLIP_CONFIG = config['paths']['path_fetalclip_config']
ARCH_NAME = 'FetalCLIP'
N_FOLDS = 5

def get_model_and_preprocess():
    # Load model configuration
    with open(PATH_FETALCLIP_CONFIG, "r") as file:
        config_fetalclip = json.load(file)
    open_clip.factory._MODEL_CONFIGS[ARCH_NAME] = config_fetalclip

    model, _, preprocess = open_clip.create_model_and_transforms(ARCH_NAME, pretrained=PATH_FETALCLIP_WEIGHT)
    model.visual.eval()
    model = model.visual
    
    model_name = ARCH_NAME

    return model_name, model, preprocess

model_name, encoder, preprocess_img = get_model_and_preprocess()
encoder.eval()
encoder = encoder.cuda()

csv_filepath = os.path.join(DIR_SAVE_CSV_EXP, f'{model_name}.csv')

dict_test_embeddings = get_test_embeddings(encoder, preprocess_img)
ds_test = DatasetPlanesDB(
    DIR_TEST, LIST_TEST_PID, preprocess_img, dict_test_embeddings,
)

# Loop through fold
for i in range(N_FOLDS):
    dict_train_embeddings, dict_val_embeddings = get_train_val_embeddings(
        encoder, DIR_TRAIN, DIR_VAL, DICT_LIST_PID[str(i)],
        preprocess_img,
    )
    ds_train = DatasetPlanesDB(
        DIR_TRAIN, DICT_LIST_PID[str(i)][0], preprocess_img, dict_train_embeddings,
    )
    ds_val = DatasetPlanesDB(
        DIR_VAL, DICT_LIST_PID[str(i)][1], preprocess_img, dict_val_embeddings,
    )
    
    print("len(train_ds)", len(ds_train))
    print("len(val_ds)", len(ds_val))
    print("len(test_ds)", len(ds_test))

    # Loop through multiple runs
    for j in range(N_RUNS_PER_EXP):
        if os.path.exists(csv_filepath):
            df_logs = pd.read_csv(csv_filepath)
            if f'{i}_{j}' in [f"{int(r['fold'])}_{int(r['run'])}" for _, r in df_logs.iterrows()]:
                continue
        
        print('*'*30)
        print('fold|||run', f'{i}|||{j}')
        print("len(train_ds)", len(ds_train))
        print("len(val_ds)", len(ds_val))
        print("len(test_ds)", len(ds_test))
        
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
        val_loader   = torch.utils.data.DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        test_loader  = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = LitModel(encoder.proj.shape[1], NUM_CLASSES)

        checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", dirpath=f"fetalclip/{i}_{j}")

        trainer = pl.Trainer(
            devices=1, accelerator='gpu', max_epochs=MAX_EPOCHS, callbacks=[checkpoint],
            check_val_every_n_epoch=CHECK_VAL_EVERY_N_EPOCH,
        )

        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, dataloaders=test_loader, ckpt_path='best')
        
        model.test_metrics = {
            'fold': i,
            'run': j,
            'n_train': len(ds_train),
            'n_val': len(ds_val),
            **{key: val.item() for key, val in model.test_metrics.items()},
        }

        df_new = pd.DataFrame([model.test_metrics])
        if os.path.exists(csv_filepath):
            df_new.to_csv(csv_filepath, mode='a', header=False, index=False)
        else:
            df_new.to_csv(csv_filepath, mode='w', header=True, index=False)