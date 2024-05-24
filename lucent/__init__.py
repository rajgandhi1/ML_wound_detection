import logging
from io import BytesIO
import os
import sys
import azure.functions as func
import requests
import numpy as np
import cv2
from PIL import Image
import torch
from pathlib import Path
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
from groundingdino.util.inference import load_model, load_image, predict, annotate

# Pre-load the GroundingDINO model if your environment supports persistent state
# Otherwise, load it on the first request and cache it if possible
dir_path = Path(__file__).parent.absolute()
dino_config = str(dir_path / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
dino_weights = str(dir_path / "GroundingDINO/weights/groundingdino_swint_ogc.pth")
model = load_model(dino_config, dino_weights)
model.eval()
device_type = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device=device_type)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get the image URL from the request
    image_url = req.params.get('imageUrl')
    if not image_url:
        return func.HttpResponse("Please pass an imageUrl in the query string", status_code=400)

    # Load the image
    response = requests.get(image_url)
    img_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return func.HttpResponse("Failed to load image from URL", status_code=400)
    logging.info('Image loaded')

    # Convert BGR to RGB (OpenCV to PIL)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Prepare the image for GroundingDINO prediction
    image_source, image = load_image(buffer)

    # Predict the boxes, logits, and phrases
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption="wound. ulcer. callus.",  # Example caption; modify as needed
            box_threshold=0.30,
            text_threshold=0.25,
            device=device_type
        )

    # Annotate the image with detections
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    logging.info('Image annotated')

    # Convert the annotated image to bytes for the response
    is_success, buffer = cv2.imencode(".png", annotated_frame)
    if not is_success:
        return func.HttpResponse("Failed to encode image", status_code=500)
    
    return func.HttpResponse(buffer.tobytes(), mimetype='image/png')
