from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Initialize FastAPI
app = FastAPI()

# Load the Keras model
try:
    MODEL = tf.keras.models.load_model("tomato.keras")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Replace these with your model’s actual class names
CLASS_NAMES = ["Healthy", "Late_Blight", "Early_Blight"]

@app.get("/ping")
async def ping():
    return {"message": "Tomato disease prediction API is running!"}

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image = image.resize((224, 224))  # <-- match your model input size
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {"class": predicted_class, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
