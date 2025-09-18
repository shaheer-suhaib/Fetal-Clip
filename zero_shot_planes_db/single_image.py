import os
import json
import torch
import numpy as np
import open_clip
from PIL import Image

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_TEXT_PROMPTS = os.path.join(script_dir, "test_five_planes_prompts.json")  # Now looks in same directory as script


class PlanesPredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.text_prompts = None
        self.text_features_list = None

    def make_image_square_with_zero_padding(self, image):
        """Convert image to square with black padding"""
        width, height = image.size
        max_side = max(width, height)
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        padding_color = (0, 0, 0) if image.mode == "RGB" else 0
        new_image = Image.new(image.mode, (max_side, max_side), padding_color)
        
        padding_left = (max_side - width) // 2
        padding_top = (max_side - height) // 2
        new_image.paste(image, (padding_left, padding_top))
        
        return new_image

    def initialize_models(self, model, tokenizer, preprocess):
        """receive model already loaded"""
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.model.to(self.device).eval()

        with open(PATH_TEXT_PROMPTS, "r") as json_file:
            self.text_prompts = json.load(json_file)

        # Precompute text features
        # Encode text prompts
        self.text_features_list = []
        with torch.no_grad():
            for prompts in self.text_prompts.values():
                tokens = self.tokenizer(prompts).to(self.device)
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.mean(dim=0, keepdim=True)
                features = features / features.norm(dim=-1, keepdim=True)
                self.text_features_list.append(features)

    def predict_image(self, image_path):    
        """Predict fetal plane for a single image"""
        # Load and preprocess image
        img = Image.open(image_path)
        img = self.make_image_square_with_zero_padding(img)
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logits = [100.0 * image_features @ features.T for features in self.text_features_list]
            probs = torch.cat(logits, dim=1).softmax(dim=-1).cpu().squeeze().numpy()
        
        # Get results
        plane_names = list(self.text_prompts.keys())
        predicted_idx = np.argmax(probs)
        predicted_plane = plane_names[predicted_idx]
        confidence = probs[predicted_idx]
        
        # Print results
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Predicted plane: {predicted_plane}")
        print(f"Confidence: {confidence:.4f}")
        print("\nAll probabilities:")
        for plane, prob in zip(plane_names, probs):
            print(f"  {plane}: {prob:.4f}")





# if __name__ == '__main__':
#     image_filename = input("Enter image filename (e.g., 'image.png'): ")
#     # Patient00166_Plane4_1_of_2.png
#     image_path = os.path.join(BASE_URL, image_filename)
    
#     if not os.path.exists(image_path):
#         print(f"Error: Image '{image_path}' not found!")
#     else:
#         predict_image(image_path)
