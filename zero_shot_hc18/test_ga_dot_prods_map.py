import os
import math
import json
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import open_clip

DIR_IMAGES = '../data/training_set'
PATH_CSV = '../data/training_set_pixel_size_and_HC.csv'
PATH_FETALCLIP_WEIGHT = "../FetalCLIP_weights.pt"
PATH_FETALCLIP_CONFIG = "../FetalCLIP_config.json"
NUM_WORKERS = 4

# Load model configuration
with open(PATH_FETALCLIP_CONFIG, "r") as file:
    config_fetalclip = json.load(file)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

INPUT_SIZE = 224
DICT_TEMPLATES = {
    "brain": [
        "Ultrasound image at {weeks} weeks and {day} days gestation focusing on the fetal brain, highlighting anatomical structures with a pixel spacing of {pixel_spacing} mm/pixel.",
        "Fetal ultrasound image at {weeks} weeks, {day} days of gestation, focusing on the developing brain, with a pixel spacing of {pixel_spacing} mm/pixel, highlighting the structures of the fetal brain.",
        "Fetal ultrasound image at {weeks} weeks and {day} days gestational age, highlighting the developing brain structures with a pixel spacing of {pixel_spacing} mm/pixel, providing important visual insights for ongoing prenatal assessments.",
        "Ultrasound image at {weeks} weeks and {day} days gestation, highlighting the fetal brain structures with a pixel spacing of {pixel_spacing} mm/pixel.",
        "Fetal ultrasound at {weeks} weeks and {day} days, showing a clear view of the developing brain, with an image pixel spacing of {pixel_spacing} mm/pixel."
    ],
}

MIN_HC = 0
MAX_HC = math.inf

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

list_ga_in_days = [weeks * 7 + days for weeks in range(14, 39) for days in range(0, 7)]
assert sorted(list_ga_in_days) == list_ga_in_days

def make_image_square_with_zero_padding(image):
    width, height = image.size

    # Determine the size of the square
    max_side = max(width, height)

    # Create a new square image with black padding (0 for black in RGB or L modes)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

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


class HCDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, path_csv, preprocess=None,sample_size = None):
        self.root_dir = root_dir
        self.preprocess = preprocess

        df = pd.read_csv(path_csv)
        df = df[df['head circumference (mm)'] >= MIN_HC]
        df = df[df['head circumference (mm)'] <= MAX_HC]
               # Limit to first N images if specified
        if sample_size is not None:
            df = df.head(sample_size)

        self.data = df.to_dict(orient='records')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]

        imagepath = os.path.join(self.root_dir, data['filename'])

        hc = data['head circumference (mm)']

        image = Image.open(imagepath)
        pixel_spacing = max(image.size) / INPUT_SIZE * data['pixel size(mm)']

        image = make_image_square_with_zero_padding(image)

        if self.preprocess:
            image = self.preprocess(image)
        
        return image, pixel_spacing, hc

def get_text_prompts(template, pixel_spacing, tokenizer, model):
    prompts = []
    for weeks in range(14, 39):
        for days in range(0, 7):
            prompt = template.replace("{weeks}", str(weeks))
            prompt = prompt.replace("{day}", str(days))
            prompt = prompt.replace("{pixel_spacing}", f"{pixel_spacing:.2f}")
            prompts.append(prompt)
    with torch.no_grad():
        prompts = tokenizer(prompts).to(device)
        text_features = model.encode_text(prompts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def get_unnormalized_dot_products(image_features, list_text_features):
    text_features = torch.cat(list_text_features, dim=0)
    text_dot_prods = (100.0 * image_features @ text_features.T)

    n_prompts = len(list_text_features)
    n_days    = len(list_text_features[0])

    # text_dot_prods = text_dot_prods.view(image_features.shape[0], n_days, n_prompts)
    # text_dot_prods = text_dot_prods.mean(dim=-1)
    text_dot_prods = text_dot_prods.view(image_features.shape[0], n_prompts, n_days)#.softmax(dim=-1)
    text_dot_prods = text_dot_prods.mean(dim=1)
    return text_dot_prods

def main(checkpoint):
    model, _, preprocess = open_clip.create_model_and_transforms("FetalCLIP", pretrained=checkpoint)
    tokenizer = open_clip.get_tokenizer("FetalCLIP")

    model.eval()
    model.to(device)

    ds = HCDataset(DIR_IMAGES, PATH_CSV, preprocess,sample_size=10)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    print("len(ds)", len(ds))

    list_exp_outs = []
    for imgs, pixel_spacing, hc in tqdm(dl):
        assert imgs.shape[0] == 1
        assert pixel_spacing.shape[0] == 1
        assert hc.shape[0] == 1

        imgs = imgs.to(device)

        with torch.no_grad():
            image_features = model.encode_image(imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            key = 'brain'
            values = DICT_TEMPLATES[key]
            values = [get_text_prompts(val, pixel_spacing[0], tokenizer, model) for val in values]
            
            text_dot_prods = get_unnormalized_dot_products(image_features, values)
            
        list_exp_outs.append({
            'true_hc'       : hc.item(),
            'pred_subview'  : key,
            'text_dot_prods': text_dot_prods.detach().cpu().numpy(),
        })
    
    torch.save(list_exp_outs, 'FetalCLIP_dot_prods_map.pt')

if __name__ == '__main__':
    main(PATH_FETALCLIP_WEIGHT)