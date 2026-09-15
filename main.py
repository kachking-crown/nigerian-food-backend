import io
import json
import numpy as np
import os
import tensorflow as tf
import tempfile
import zipfile
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

# --- 1. SETUP FASTAPI ---
app = FastAPI(title="Nigerian Food Health Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LOAD THE MODEL ---
def load_model_compatibly(model_path):
    with zipfile.ZipFile(model_path, "r") as source:
        config = json.loads(source.read("config.json"))

        def remove_incompatible_fields(value):
            if isinstance(value, dict):
                value.pop("quantization_config", None)
                for child in value.values():
                    remove_incompatible_fields(child)
            elif isinstance(value, list):
                for child in value:
                    remove_incompatible_fields(child)

        remove_incompatible_fields(config)

        temporary_model = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
        temporary_model.close()
        try:
            with zipfile.ZipFile(temporary_model.name, "w") as target:
                for item in source.infolist():
                    content = json.dumps(config).encode("utf-8") if item.filename == "config.json" else source.read(item.filename)
                    target.writestr(item, content)
            return tf.keras.models.load_model(temporary_model.name, compile=False)
        finally:
            os.unlink(temporary_model.name)


print("Loading model...")
model = load_model_compatibly('nigerian_food_model.keras')
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
print("Model loaded successfully!")

# --- 3. DEFINE CLASSES AND NUTRITION DATA ---
class_names = [
    'abacha', 'afang soup', 'akara', 'banga soup', 'edikaikong soup', 
    'egusi soup', 'ewedu soup', 'fried rice', 'jollof rice', 
    'ogbono soup', 'okra soup', 'pounded yam', 'suya'
]

nutrition_db = {
    "pounded yam": {"calories": 250, "fat": 0.5, "protein": 2},
    "fried rice": {"calories": 380, "fat": 12, "protein": 8},
    "edikaikong soup": {"calories": 200, "fat": 15, "protein": 12},
    "ewedu soup": {"calories": 100, "fat": 5, "protein": 4},
    "okra soup": {"calories": 120, "fat": 6, "protein": 5},
    "afang soup": {"calories": 250, "fat": 20, "protein": 10},
    "banga soup": {"calories": 280, "fat": 22, "protein": 9},
    "abacha": {"calories": 300, "fat": 15, "protein": 6},
    "egusi soup": {"calories": 300, "fat": 25, "protein": 14},
    "ogbono soup": {"calories": 310, "fat": 24, "protein": 8},
    "akara": {"calories": 200, "fat": 12, "protein": 8},
    "jollof rice": {"calories": 350, "fat": 10, "protein": 7},
    "suya": {"calories": 250, "fat": 15, "protein": 20}
}

portion_multipliers = {
    "Small": 0.6,
    "Medium": 1.0,
    "Large": 1.5
}

def evaluate_health(cal, fat):
    if cal < 250 and fat < 15:
        return "✅ Healthy"
    elif cal < 400 and fat < 25:
        return "⚠️ Moderate (Eat in moderation)"
    else:
        return "❌ Unhealthy (High calories/fat)"

# --- 4. THE API ENDPOINT ---
@app.get("/")
def read_root():
    return {"message": "Nigerian Food Health Scanner API is running!"}

@app.post("/scan")
async def scan_food(
    file: UploadFile = File(...), 
    portion: str = Form("Medium")
):
    try:
        # Read and process the image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
        
        # Preprocess
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make Prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        food_name = class_names[predicted_class_index]
        
        # Get Nutrition and Apply Portion Multiplier
        multiplier = portion_multipliers.get(portion, 1.0)
        base = nutrition_db.get(food_name)
        
        if base:
            cal = base["calories"] * multiplier
            fat = base["fat"] * multiplier
            protein = base["protein"] * multiplier
            verdict = evaluate_health(cal, fat)
            
            # Return the result to the Flutter app
            return {
                "status": "success",
                "food": food_name.title(),
                "confidence": f"{confidence:.2f}%",
                "portion": portion,
                "calories": f"{cal:.0f} kcal",
                "fat": f"{fat:.1f}g",
                "protein": f"{protein:.1f}g",
                "verdict": verdict
            }
        else:
            return {"status": "error", "message": "Food not found in database"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}