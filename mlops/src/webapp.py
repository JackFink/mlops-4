import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from src.inference.inference import load_checkpoint, init_model, inference
from src.postprocess.postprocess import read_db


from fastapi.staticfiles import StaticFiles

# Get the absolute path to the project's root directory (mlops)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")

app = FastAPI()
app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

load_dotenv()
checkpoint = load_checkpoint()
model, scaler, label_encoder = init_model(checkpoint)


class InferenceInput(BaseModel):
    user_id: int
    content_id: int
    watch_seconds: int
    rating: float
    popularity: float


@app.post("/predict")
async def predict(input_data: InferenceInput):
    try:
        data = np.array([
            input_data.user_id,
            input_data.content_id,
            input_data.watch_seconds,
            input_data.rating,
            input_data.popularity
        ])
        recommend = inference(model, scaler, label_encoder, data)
        recommend = [int(r) for r in recommend]
        return {"recommended_content_id": recommend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
        
@app.get("/batch-predict")
async def batch_predict(k: int = 5):
    try:
        recommend = read_db("mlops", "recommend", k=k)
        return {"recommended_content_id": recommend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)