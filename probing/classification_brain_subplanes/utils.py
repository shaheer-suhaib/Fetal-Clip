import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import albumentations as A

from torchmetrics import Accuracy, F1Score
from PIL import Image

with open('config.json', 'r') as file:
    config = json.load(file)

DIR_TRAIN = os.path.join(config['paths']['dir_preprocessed'], 'train')
DIR_VAL = os.path.join(config['paths']['dir_preprocessed'], 'val')
DIR_TEST = os.path.join(config['paths']['dir_preprocessed'], 'test')

PATH_TRAIN_VAL_SPLIT = config['paths']['path_train_val_split']
PATH_TEST_SPLIT = config['paths']['path_test_split']
DIR_SAVE_CSV_EXP = config['paths']['dir_experiment_logs']

NUM_WORKERS = config['params']['num_workers']
BATCH_SIZE = config['params']['batch_size']
MAX_EPOCHS = config['params']['max_epochs']

IMG_SIZE = 224
N_RUNS_PER_EXP = 5
CHECK_VAL_EVERY_N_EPOCH = 1
PIN_MEMORY = True

DICT_CLSNAME_TO_CLSINDEX = {
    'Trans-thalamic': 0,
    'Trans-cerebellum': 1,
    'Trans-ventricular': 2,
}

with open(PATH_TRAIN_VAL_SPLIT, 'r') as file:
    DICT_LIST_PID = json.load(file)

with open(PATH_TEST_SPLIT, 'r') as file:
    LIST_TEST_PID = json.load(file)

NUM_CLASSES  = len(DICT_CLSNAME_TO_CLSINDEX)

os.makedirs(DIR_SAVE_CSV_EXP, exist_ok=True)

class DatasetPlanesDB(torch.utils.data.Dataset):
    def __init__(
            self,
            root,
            list_patients,
            preprocess_img,
            dict_embeddings=None,
        ):
        
        self.preprocess_img = preprocess_img
        self.dict_embeddings = dict_embeddings

        self.data = []
        for filename in os.listdir(root):
            pid = filename.split('_')[0]
            if pid not in list_patients:
                continue

            if self.dict_embeddings:
                self.data.append((
                    os.path.join(root, filename),
                    dict_embeddings[filename.split('.')[0]],
                ))
            else:
                self.data.append(os.path.join(root, filename))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.dict_embeddings:
            data = np.load(self.data[index][0])
            embs = self.data[index][1]
        else:
            data = np.load(self.data[index])

        img = data['img']
        ann = str(data['label'])
        
        img = Image.fromarray(img)

        img = self.preprocess_img(img)
        ann = DICT_CLSNAME_TO_CLSINDEX[ann]
        
        if self.dict_embeddings:
            return img, ann, embs
        else:
            return img, ann

class LitModel(pl.LightningModule):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.head = torch.nn.Linear(input_dim, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []
    
    def forward(self, x):
        x = self.head(x)
        return x

    def training_step(self, batch, batch_nb):
        _, y, embs = batch

        pred = self.forward(embs)
        
        loss = self.criterion(pred, y)
        pred = torch.argmax(pred, 1)
        acc = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(pred.to('cpu'), y.to('cpu')).item()
        f1  = F1Score(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(pred.to('cpu'), y.to('cpu')).item()

        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_nb):
        _, y, embs = batch

        pred = self.forward(embs)
        
        self.validation_step_outputs.append((pred,y))
        return pred, y
    
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def on_validation_epoch_end(self):
        preds = []
        targets = []

        for outs in self.validation_step_outputs:
            preds.append(outs[0])
            targets.append(outs[1])
        
        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()

        loss = self.criterion(preds, targets)

        acc = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(preds,targets)
        f1 = F1Score(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(preds,targets)

        acc_top2 = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=2, average='macro')(preds,targets)
        acc_top3 = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=3, average='macro')(preds,targets)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc_top2', acc_top2, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc_top3', acc_top3, on_step=False, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        preds = []
        targets = []

        for outs in self.validation_step_outputs:
            preds.append(outs[0])
            targets.append(outs[1])
        
        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()

        loss = self.criterion(preds, targets)

        macc = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(preds,targets)
        mf1 = F1Score(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(preds,targets)

        macc_top2 = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=2, average='macro')(preds,targets)

        self.test_metrics = {
            'test_loss': loss,
            'test_f1': mf1,
            'test_acc': macc,
            'test_acc_top2': macc_top2,
        }

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_f1', mf1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', macc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc_top2', macc_top2, on_step=False, on_epoch=True, prog_bar=True)

        acc = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1, average='none')(preds,targets)
        f1 = F1Score(task="multiclass", num_classes=self.num_classes, top_k=1, average='none')(preds,targets)

        # long_string = f"{mf1.item()},{macc.item()},{macc_top2.item()},{','.join([str(x.item()) for x in f1])},{','.join([str(x.item()) for x in acc])},{','.join([str(x.item()) for x in acc_top2])}"
        
        for key, val in DICT_CLSNAME_TO_CLSINDEX.items():
            self.test_metrics[f'test_acc_{key}'] = acc[val]
            self.test_metrics[f'test_f1_{key}']  =  f1[val]

        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.head.parameters(), lr=3e-4, weight_decay=0.01)