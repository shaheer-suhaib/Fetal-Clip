import os
import json
import torch
import numpy as np
import open_clip
from PIL import Image

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

PATH_TEXT_PROMPTS = os.path.join(script_dir, "test_brain_subplanes_prompts.json")  # Now looks in same directory as script


# ---------------- UTILS ----------------


class BrainPlanesPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.list_text_features = None
        self.class_names = None

    def make_image_square_with_zero_padding(self, image):
        width, height = image.size
        max_side = max(width, height)

        if image.mode == "RGB":
            padding_color = (0, 0, 0)
        elif image.mode == "L":
            padding_color = 0
        else:
            image = image.convert("RGB")
            padding_color = (0, 0, 0)

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
            text_prompts = json.load(json_file)

        # Precompute text features
        list_text_features = []
        with torch.no_grad():
            for plane, prompts in text_prompts.items():
                tokens = self.tokenizer(prompts).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.mean(dim=0).unsqueeze(0)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                list_text_features.append(text_features)
        self.list_text_features = torch.cat(list_text_features, dim=0)
        self.class_names = list(text_prompts.keys())

    def predict_image(self, image_path):
        try:
            img = Image.open(image_path)
            img = self.make_image_square_with_zero_padding(img)

            if img.mode != "RGB":
                img = img.convert("RGB")

            tensor_img = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(tensor_img)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                logits = 100.0 * image_features @ self.list_text_features.T
                probs = logits.softmax(dim=-1).cpu().numpy()[0]

            max_idx = int(np.argmax(probs))
            pred_class = self.class_names[max_idx]
            pred_prob = float(probs[max_idx])

            return {
                "predicted_class": pred_class,
                "probability": round(pred_prob, 4),
                "all_probs": {cls: float(prob) for cls, prob in zip(self.class_names, probs)}
            }

        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
                PATH_FETALCLIP_CONFIG =  "D:\DE 44 CE 2022\mySemData\sem 7\FYP\FetalClip\FetalCLIP\FetalCLIP_config.json"
                PATH_FETALCLIP_WEIGHT = "D:\DE 44 CE 2022\mySemData\sem 7\FYP\FetalClip\FetalCLIP\FetalCLIP_weights.pt"

                # Load model configuration
                with open(PATH_FETALCLIP_CONFIG, "r") as file:
                    config_fetalclip = json.load(file)
                open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    "FetalCLIP", pretrained=PATH_FETALCLIP_WEIGHT
                )
                tokenizer = open_clip.get_tokenizer("FetalCLIP")

                predictor = BrainPlanesPredictor()
                predictor.initialize_models(model, tokenizer, preprocess)

                image_path = r"D:\DE 44 CE 2022\mySemData\sem 7\FYP\FetalClip\FetalCLIP\data\Planes\FETAL_PLANES_ZENODO\Images\Patient00168_Plane3_1_of_3.png" 
                result = predictor.predict_image(image_path)
                print(result)
