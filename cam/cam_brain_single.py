import os
import json
import torch
import numpy as np
from PIL import Image
import open_clip
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_FETALCLIP_CONFIG = "../FetalCLIP_config.json"
PATH_TEXT_PROMPTS = os.path.join(script_dir, "test_brain_subplanes_prompts.json")


class BrainSubplanesCAMPredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.text_prompts = None
        self.text_features = None
        self.model_wrapper = None
        self.cam = None

    def reshape_transform(self, tensor, height=16, width=16):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def make_image_square_with_zero_padding(self, image):
        width, height = image.size
        max_side = max(width, height)

        if image.mode == "RGBA":
            image = image.convert("RGB")

        padding_color = (0, 0, 0) if image.mode == "RGB" else 0
        new_image = Image.new(image.mode, (max_side, max_side), padding_color)
        padding_left = (max_side - width) // 2
        padding_top = (max_side - height) // 2
        new_image.paste(image, (padding_left, padding_top))
        return new_image

    def get_text_features(self):
        all_text_features = []
        for prompts in self.text_prompts.values():
            tokens = self.tokenizer(prompts).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.mean(dim=0).unsqueeze(0)
            features = features / features.norm(dim=-1, keepdim=True)
            all_text_features.append(features)
        return torch.cat(all_text_features, dim=0)

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, text_features, device):
            super().__init__()
            self.model = model
            self.text_features = text_features
            self.device = device

        def forward(self, imgs):
            imgs = imgs.to(self.device)
            image_features = self.model.encode_image(imgs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_logits = 100.0 * image_features @ self.text_features.T
            text_probs = text_logits.softmax(dim=-1)
            return text_probs

    def initialize_models(self, model, tokenizer, preprocess):
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.preprocess = preprocess

        # load text prompts
        with open(PATH_TEXT_PROMPTS, "r") as f:
            self.text_prompts = json.load(f)

        # precompute text features
        self.text_features = self.get_text_features()

        # wrap model for CAM
        self.model_wrapper = self.ModelWrapper(self.model, self.text_features, self.device).to(self.device).eval()
        target_layers = [self.model_wrapper.model.visual.transformer.resblocks[-1].ln_1]
        self.cam = GradCAM(model=self.model_wrapper, target_layers=target_layers, reshape_transform=self.reshape_transform)

    def predict_image(self, image_path):
        try:
            img = Image.open(image_path)
            img = self.make_image_square_with_zero_padding(img)
            if img.mode != "RGB":
                img = img.convert("RGB")

            img_for_cam = img.resize((224, 224))
            rgb_img = np.array(img_for_cam).astype(np.float32) / 255.0
            if len(rgb_img.shape) == 2:
                rgb_img = np.stack([rgb_img] * 3, axis=-1)

            tensor_img = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds = self.model_wrapper(tensor_img)[0]
                max_class_idx = preds.argmax().item()
                max_class = list(self.text_prompts.keys())[max_class_idx]
                max_prob = preds[max_class_idx].item()

            targets = [ClassifierOutputTarget(max_class_idx)]
            grayscale_cam = self.cam(input_tensor=tensor_img, targets=targets)[0, :]
            grayscale_cam = np.maximum(grayscale_cam, 0)
            if grayscale_cam.max() > 0:
                grayscale_cam = grayscale_cam / grayscale_cam.max()

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.5)
            return f"Predicted: {max_class} ({max_prob:.2f})", Image.fromarray(cam_image)

        except Exception as e:
            return f"Error during prediction: {str(e)}", None
