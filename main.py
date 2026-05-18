from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Crop Disease Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

def get_treatment(disease_name: str) -> str:
    clean_name = disease_name.replace("___", " — ").replace("_", " ")
    prompt = f"""You are an agricultural expert. A farmer's crop has been diagnosed with: {clean_name}.
Give exactly 3 sentences:
1. What this disease is
2. The most effective treatment
3. A prevention tip for the future
Be specific, practical, and use simple language a farmer would understand."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()

    try:
        from model.predict import predict
        result = predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    treatment = get_treatment(result["predicted_class"])

    return {
        "disease": result["predicted_class"].replace("___", " — ").replace("_", " "),
        "confidence": result["confidence"],
        "top3": result["top3"],
        "treatment": treatment
    }

@app.get("/health")
def health():
    return {"status": "ok"}