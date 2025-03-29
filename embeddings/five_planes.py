import json
import os

import torch
import pandas as pd
import open_clip
from PIL import Image
from tqdm import tqdm

# Constants and Configuration
ROOT_DIR = "<path_to_fetal_planes_db_data>/Images"
PATH_CSV = "<path_to_fetal_planes_db_data>/FETAL_PLANES_DB_data.csv"
PATH_FETALCLIP_WEIGHT = "../FetalCLIP_weights.pt"
PATH_FETALCLIP_CONFIG = "../FetalCLIP_config.json"
BATCH_SIZE = 16
NUM_WORKERS = 4

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
    'Other': 'other',
}


class DatasetFetalPlanesDB(torch.utils.data.Dataset):
    """
    We exclude 'Other'
    """
    def __init__(
            self, root_dir, path_csv, preprocess, split='all',
            exclude_planes=['Other'],
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
        # print(img.size)
        img = make_image_square_with_zero_padding(img)
        # print(img.size)

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

model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained=PATH_FETALCLIP_WEIGHT)
model.eval()
model.to(device)

ds = DatasetFetalPlanesDB(ROOT_DIR, PATH_CSV, preprocess, split='all', exclude_planes=[])
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

list_paths = []
list_gt = []
list_image_features = []
for imgs, planes, paths in tqdm(dl):
    with torch.no_grad(), torch.cuda.amp.autocast():
        imgs = imgs.to(device)
        image_features = model.encode_image(imgs)
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        
    list_paths.extend(paths)
    list_gt.extend(planes)
    list_image_features.append(image_features.detach().cpu())

image_features = torch.cat(list_image_features, dim=0).float()

targets = [plane_db_to_planename[x] for x in list_gt]

torch.save(
    {
        'plane_db_to_planename': plane_db_to_planename,
        'targets': targets,
        'image_features': image_features.detach().cpu(),
        'paths': list_paths,
    },
    os.path.join('FetalCLIP_embs_five_planes.pt')
)
