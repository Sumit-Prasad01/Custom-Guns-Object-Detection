import io 
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
from config.paths_config import *
from utils.helpers import predict_and_draw

app = FastAPI()

@app.get("/")
def read_root():
    return {"message" : "Welcome to the Guns Object Detection API"}


@app.post("/predict/")
async def predict(file:UploadFile=File(...)):

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    output_image = predict_and_draw(image)

    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr , format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr , media_type="image/png")


def main():
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)

if __name__ == "__main__":
    main()