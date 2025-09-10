import json
import torch
import open_clip
from PIL import Image
import os

# Define paths for model configuration and weights
PATH_FETALCLIP_CONFIG = "FetalCLIP_config.json"
PATH_FETALCLIP_WEIGHT = "FetalCLIP_weights.pt"
PATH_IMAGE_DIR = r"D:\DE 44 CE 2022\mySemData\sem 7\FYP\FetalClip\FetalCLIP\example_images"

# Automatically detect the best available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and register model configuration
with open(PATH_FETALCLIP_CONFIG, "r") as file:
    config_fetalclip = json.load(file)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

# Load the FetalCLIP model and preprocessing transforms as well as tokenizer
model, preprocess_train, preprocess_test = open_clip.create_model_and_transforms("FetalCLIP", pretrained=PATH_FETALCLIP_WEIGHT)
tokenizer = open_clip.get_tokenizer("FetalCLIP")
model.eval()
model.to(device)

# Select the images you want to process
# image_files = ["abdominal.png", "brain1.png"]

image_files = [
    "3VV.tiff", "4CH.tiff", "ABDOMINAL - Copy.tiff", "abdominal.png", 
    "ABDOMINAL.tiff", "abdominal1.png", "abdominal2(yt).png", "BRAIN-CB.tiff", 
    "BRAIN-TV.tiff", "brain1 - Copy.png", "brain1.png", "brain2.png", 
    "brain3(yt).png", "femur.png", "FEMUR.tiff", "FEMUR(1).jpg", 
    "femur(yt).png", "femur1.png", "FEMURedited.png", "FEMURedited2.png", 
    "FEMURedited3.png", "image.png", "KIDNEYS.tiff", "LIPS.tiff", 
    "LVOT.tiff", "PROFILE.tiff", "RVOT.tiff", "SPINE-CORONAL.tiff", 
    "SPINE-SAGITTAL.tiff"
]
images_paths = [os.path.join(PATH_IMAGE_DIR, img) for img in image_files]

# Verify images exist
for img_path in images_paths:
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")

# Preprocess images and stack them into a single tensor
images = torch.stack([preprocess_test(Image.open(img_path)) for img_path in images_paths]).to(device)

# Define comprehensive text prompts for classification
text_prompts = [
    "Ultrasound image focusing on the fetal abdominal area, highlighting structural development.",
    "Fetal ultrasound image focusing on the heart, highlighting detailed cardiac structures.",
    "Fetal brain ultrasound showing neural development and brain structures.",
    "Ultrasound image of fetal spine and vertebral column.",
    "Fetal ultrasound showing limb and extremity development.",
    "Ultrasound image of fetal thoracic cavity and chest area."
]

# Create labels for better interpretation
labels = ["Abdominal", "Heart", "Brain", "Spine", "Limbs", "Thorax"]

print(f"Processing {len(images)} images with {len(text_prompts)} text prompts...")

# Tokenize the text prompts
text_tokens = tokenizer(text_prompts).to(device)

# Perform model inference
with torch.no_grad():
    # Use autocast only if CUDA is available
    if device.type == 'cuda':
        with torch.cuda.amp.autocast():
            text_features = model.encode_text(text_tokens)
            image_features = model.encode_image(images)
    else:
        # No autocast for CPU
        text_features = model.encode_text(text_tokens)
        image_features = model.encode_image(images)

    # Normalize feature vectors
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity scores (probabilities) between image and text features
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Display results in a more readable format
print("\n" + "="*60)
print("FETALCLIP CLASSIFICATION RESULTS")
print("="*60)

for i, (img_file, img_probs) in enumerate(zip(image_files, text_probs)):
    print(f"\nImage: {img_file}")
    print("-" * 40)
    
    # Get top 3 predictions
    top_indices = torch.topk(img_probs, k=min(3, len(labels))).indices
    
    for j, idx in enumerate(top_indices):
        probability = img_probs[idx].item() * 100
        print(f"  {j+1}. {labels[idx]}: {probability:.2f}%")
    
    # Show all probabilities
    print(f"\nAll probabilities:")
    for label, prob in zip(labels, img_probs):
        print(f"  {label}: {prob.item()*100:.2f}%")

print("\n" + "="*60)