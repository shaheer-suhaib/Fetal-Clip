import json
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import open_clip

from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROOT_DIR = "../data/Planes/FETAL_PLANES_ZENODO/Images"
PATH_CSV = "../data/Planes/FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv"
PATH_FETALCLIP_WEIGHT = "../FetalCLIP_weights.pt"
PATH_FETALCLIP_CONFIG = "../FetalCLIP_config.json"
PATH_TEXT_PROMPTS = "test_five_planes_prompts.json"
SAVE_DIR = 'CAM_5_planes'
SAVE_DIR_ORI_IMG = os.path.join(SAVE_DIR, 'Image')

BATCH_SIZE = 1
NUM_WORKERS = 4
N_SAMPLES = 10

TARGET_LABELS = [
    'Fetal brain',
    'Fetal abdomen',
    'Fetal thorax',
    'Fetal femur',
    'Maternal cervix',
]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model configuration
with open(PATH_FETALCLIP_CONFIG, "r") as file:
    config_fetalclip = json.load(file)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

plane_db_to_planename = {
    'Fetal brain': 'brain',
    'Fetal abdomen': 'abdomen',
    'Fetal thorax': 'heart',
    'Fetal femur': 'femur',
    'Maternal cervix': 'cervix',
}

class DatasetFetalPlanesDB(torch.utils.data.Dataset):
    """
    We exclude 'Other'
    """
    def __init__(
            self, root_dir, path_csv, preprocess, split='all',
            exclude_planes=['Other'], sample_size=None
        ):
        self.root = root_dir
        self.preprocess = preprocess
        
        df = pd.read_csv(path_csv, sep=';')
        if split == 'train':
            df = df[df['Train '] == 1]
        elif split == 'test':
            df = df[df['Train '] == 0]
        
        for plane in exclude_planes:
            df = df[df['Plane'] != plane]
          # Limit sample size if specified
        if sample_size is not None:
            df = df.head(sample_size)
        
        self.data = []
        for _, row in df.iterrows():
            self.data.append({
                'img': os.path.join(self.root, f"{row['Image_name']}.png"),
                'plane': row['Plane'],
                'pid': row['Patient_num'],
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = Image.open(self.data[index]['img'])
        img = make_image_square_with_zero_padding(img)
        
        if self.preprocess:
            img = self.preprocess(img)
        plane = self.data[index]['plane']
        
        return img, plane, self.data[index]['img']

def make_image_square_with_zero_padding(image):
    width, height = image.size

    # Determine the size of the square
    max_side = max(width, height)

    # Create a new square image with black padding (0 for black in RGB or L modes)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Create a new square image with black padding (0 for black in RGB or L modes)
    if image.mode == "RGB":
        padding_color = (0, 0, 0)  # Black for RGB images
    elif image.mode == "L":
        padding_color = 0  # Black for grayscale images

    # Create a new square image
    new_image = Image.new(image.mode, (max_side, max_side), padding_color)

    # Calculate padding
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2

    # Paste the original image in the center of the new square image
    new_image.paste(image, (padding_left, padding_top))

    return new_image

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR_ORI_IMG, exist_ok=True)

with open(PATH_TEXT_PROMPTS, 'r') as json_file:
    text_prompts = json.load(json_file)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, preprocess, tokenizer):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
    
    def forward(self, imgs):
        list_text_features = []
        for plane, prompts in tqdm(text_prompts.items()):
            prompts = self.tokenizer(prompts).to(device)
            text_features = self.model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            #
            text_features = text_features.mean(dim=0).unsqueeze(0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            #
            list_text_features.append(text_features)
        
        imgs = imgs.to(device)
        image_features = self.model.encode_image(imgs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        list_text_logits = []
        for text_features in list_text_features:
            text_logits = (100.0 * image_features @ text_features.T).mean(dim=-1)[:,None]
            list_text_logits.append(text_logits)
        text_probs = torch.cat(list_text_logits, dim=1).softmax(dim=-1)
        
        return text_probs

model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained=PATH_FETALCLIP_WEIGHT)
tokenizer = open_clip.get_tokenizer("FetalCLIP")

ds = DatasetFetalPlanesDB(ROOT_DIR, PATH_CSV, preprocess, split='all', exclude_planes=['Other'],sample_size=3)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model_wrapper = ModelWrapper(model, preprocess, tokenizer)
model_wrapper.eval()
model_wrapper.to(device)

target_layers = [model_wrapper.model.visual.transformer.resblocks[-1].ln_1]

def reshape_transform(tensor, height=16, width=16):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

cam = ScoreCAM(
    model=model_wrapper, target_layers=target_layers,
    reshape_transform=reshape_transform
)

for target_label in TARGET_LABELS:
    curr_n = 0
    for tensor_img, label, imgpath in tqdm(ds):
        model_wrapper.zero_grad()
        if label != target_label:
            continue
        
        curr_n += 1
        imgname = os.path.basename(imgpath).split('.')[0]

        rgb_img = tensor_img.numpy().transpose(1,2,0)
        rgb_img = (rgb_img - rgb_img.min((0,1))) / (rgb_img.max((0,1)) - rgb_img.min((0,1)))

        tensor_img = tensor_img.unsqueeze(0)

        targets = list(text_prompts.keys()).index(plane_db_to_planename[label])
        targets = [ClassifierOutputTarget(targets)]
        grayscale_cam = cam(input_tensor=tensor_img, targets=targets,)

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False, image_weight=0.5)

        with torch.no_grad():
            pred = model_wrapper(tensor_img)
            assert len(pred) == 1
            pred = pred[0]
            max_class = pred.argmax()
            max_prob = pred[max_class].cpu().detach().item()
            max_class = max_class.cpu().detach().item()
        
        max_class = list(text_prompts.keys())[max_class]
        cv2.imwrite(os.path.join(SAVE_DIR, f"{plane_db_to_planename[label]}_-_{imgname}_-_{max_class}_-_{int(max_prob*100)}.png"), cam_image)

        shutil.copy(
            imgpath,
            os.path.join(SAVE_DIR_ORI_IMG, f"{plane_db_to_planename[label]}_-_{os.path.basename(imgpath)}"),
        )

        if plane_db_to_planename[label] != max_class:
            print(plane_db_to_planename[label], max_class, max_prob)

        if curr_n == N_SAMPLES:
            break