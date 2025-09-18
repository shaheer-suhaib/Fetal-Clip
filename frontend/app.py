
# import gradio as gr
# import os
# import sys
# from PIL import Image

# # Add the parent directory to Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

# # Import prediction functions
# try:
#     from zero_shot_planes_db.single_image import predict_image
#     from cam.cam_new import predict_image as cam_image
#     print("Successfully imported prediction functions")
# except ImportError as e:
#     print(f"Import error: {e}")
#     def predict_image(image_path):
#         return f"Import failed - function not available. Error: {e}"
#     def cam_image(image_path):
#         return None

# def predict_from_gradio(file):
#     """Run both normal prediction and CAM visualization"""
#     if file is None:
#         return "No file uploaded", None


#     try:
#         # Verify it's an image
#         img = Image.open(file)
#         img.verify()
#         img = Image.open(file)

#         # Run classification prediction
#         import io, contextlib
#         output_buffer = io.StringIO()
#         with contextlib.redirect_stdout(output_buffer):
#             result = predict_image(file)
#         captured_output = output_buffer.getvalue()

#         if captured_output.strip():
#             pred_text = captured_output
#         elif result:
#             pred_text = str(result)
#         else:
#             pred_text = "Prediction completed successfully"

#         # Run CAM heatmap
#         cam_result = cam_image(file)  # should return path or PIL image

#         return pred_text, cam_result

#     except Exception as e:
#         return f"Error: {str(e)}", None


# # Gradio interface with 3 outputs (original + prediction + CAM)
# demo = gr.Interface(
#     fn=predict_from_gradio,
#     inputs=gr.Image(type="filepath", label="Upload Ultrasound Image"),  # <-- change here
#     outputs=[
       
#         gr.Textbox(label="Prediction Result"),    # <-- text result
#         gr.Image(label="CAM Visualization")       # <-- CAM heatmap
#     ],
#     title="FetalCLIP Predictor with CAM"
# )


# if __name__ == "__main__":
#     demo.launch()

# brain_subplanes_single_image.py

import open_clip
import gradio as gr
import os
import sys,json
from PIL import Image



# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
PATH_FETALCLIP_CONFIG =  "../FetalCLIP_config.json"
PATH_FETALCLIP_WEIGHT = "../FetalCLIP_weights.pt"


# Import prediction functions
try:
    # from zero_shot_planes_db.single_image import predict_image as predict_five
    # from zero_shot_planes_db.brain_subplanes_single_image import predict_image as predict_brain
    # from cam.cam_new import predict_image as cam_image
    # from cam.cam_brain_single import predict_image as cam_brain_image
    from cam.cam_brain_single import BrainSubplanesCAMPredictor
    from cam.cam_new import CAMPredictor
    from zero_shot_planes_db.single_image import PlanesPredictor
    from zero_shot_planes_db.brain_subplanes_single_image import BrainPlanesPredictor
    


     # Load model configuration
    with open(PATH_FETALCLIP_CONFIG, "r") as file:
        config_fetalclip = json.load(file)
    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "FetalCLIP", pretrained=PATH_FETALCLIP_WEIGHT
    )
    tokenizer = open_clip.get_tokenizer("FetalCLIP")


    #initializing models
    cam_brain_image = BrainSubplanesCAMPredictor()
    cam_image=CAMPredictor()
    predict_five =    PlanesPredictor() 
    predict_brain = BrainPlanesPredictor()

    cam_brain_image.initialize_models(model, tokenizer, preprocess)
    predict_five.initialize_models(model, tokenizer, preprocess)
    predict_brain.initialize_models(model, tokenizer, preprocess)
    cam_image.initialize_models(model, tokenizer, preprocess)   

    
    print("Successfully imported prediction functions")
except ImportError as e:
    print(f"Import error: {e}")
    def predict_brain(image_path): return f"Brain model import failed: {e}"
    def predict_five(image_path): return f"Five-planes model import failed: {e}"
    def cam_image(image_path): return None


def predict_from_gradio(file, mode):
    """Run selected prediction (brain / five planes) + CAM visualization"""
    if file is None:
        return "No file uploaded", None

    try:
        # verify image
        img = Image.open(file)
        img.verify()
        img = Image.open(file)

        # choose model based on dropdown
        if mode == "Brain Planes":
            predict_fn = predict_brain
        else:
            predict_fn = predict_five

        # run classification
        import io, contextlib
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            result = predict_fn.predict_image(file)
        captured_output = output_buffer.getvalue()

        if captured_output.strip():
            pred_text = captured_output
        elif result:
            pred_text = str(result)
        else:
            pred_text = "Prediction completed successfully"

        # run CAM
        if mode == "Brain Planes":
            cam_result = cam_brain_image.predict_image(file)
        else:
            cam_result = cam_image.predict_image(file)

        return pred_text, cam_result

    except Exception as e:
        return f"Error: {str(e)}", None


# gradio interface
demo = gr.Interface(
    fn=predict_from_gradio,
    inputs=[
        gr.Image(type="filepath", label="Upload Ultrasound Image"),
        gr.Dropdown(
            ["Brain Planes", "Five Planes"],
            label="Select Prediction Mode",
            value="Brain Planes"
        ),
    ],
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Image(label="CAM Visualization"),
    ],
    title="FetalCLIP Predictor with CAM"
)


if __name__ == "__main__":
    demo.launch()