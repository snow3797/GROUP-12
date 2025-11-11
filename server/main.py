import os
import shutil
import string
import random
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.cnn_model_load import predict_leaf_class
from api.ART import ask_openai
from api.DB_integration import insert_metadata

# -------------------------------------------------
# ⚙️ FastAPI setup
# -------------------------------------------------
app = FastAPI(title="🌽 Maize Disease Detection API")

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Directory to store uploaded images
METADATA_DIR = "api/metadata"
os.makedirs(METADATA_DIR, exist_ok=True)

# -------------------------------------------------
# 🔡 Helper function: generate unique filename
# -------------------------------------------------
def generate_random_name(length=14):
    """Generate a random string for naming saved images."""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

# -------------------------------------------------
# 🚀 Serve the frontend
# -------------------------------------------------
@app.get("/")
async def serve_frontend():
    return FileResponse(r"frontend\index.html")

# -------------------------------------------------
# 🚀 Main API endpoint
# -------------------------------------------------
@app.post("/analyze/")
async def analyze_leaf(image: UploadFile = File(...)):
    """
    Accepts an image file, runs disease prediction,
    queries OpenAI for an explanation, stores metadata,
    and returns both prediction + AI response.
    """
    try:
        # 1️⃣ Save uploaded image locally
        filename = generate_random_name() + os.path.splitext(image.filename)[1]
        save_path = os.path.join(METADATA_DIR, filename)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        print(f"[INFO] Image saved: {save_path}")

        # 2️⃣ Run prediction using trained CNN model
        pred_disease = predict_leaf_class(save_path)
        print(f"[INFO] Predicted disease: {pred_disease}")

        # 3️⃣ Ask OpenAI for explanation
        prompt = (
            f"Explain the maize disease '{pred_disease}' in under 250 words. "
            f"Include causes, visible symptoms, effects on crops, and recommended treatments."
        )
        try:
            openai_response = ask_openai(prompt)
            print(openai_response)
            print(f"[INFO] OpenAI Response Received.")
        except Exception as e:
            print(f"[WARN] OpenAI request failed: {e}")
            openai_response = "Unable to fetch AI explanation at this time."

        # 4️⃣ Store prediction + OpenAI result in SQL Server
        try:
            insert_metadata(
                user_id=filename,
                imagedirectory=save_path,
                openai_response=openai_response,
                diagnosis=pred_disease
            )
        except Exception as db_err:
            print(f"[WARN] Database insert failed: {db_err}")

        # 5️⃣ Return response to frontend
        return JSONResponse(
            content={
                "status": "success",
                "image_name": filename,
                "predicted_disease": pred_disease,
                "openai_response": openai_response
            },
            status_code=200
        )

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Maize Disease Detection API is running"}

# -------------------------------------------------
# 🧩 Entry point for local testing
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )