from fastapi import FastAPI
from src.speech_to_text import on_turn

app = FastAPI()

@app.post("/interview")
def run_interview():
    data = on_turn()
    return {"status": "success", "data": data}

@app.get("/health")
def health():
    return {"status": "healthy"}