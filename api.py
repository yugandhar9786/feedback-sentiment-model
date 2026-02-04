from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

class TextInput(BaseModel):
    text: str

# Load model
model_path = "./model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

labels = ["negative", "positive"]

@app.post("/predict")
def predict_sentiment(data: TextInput):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return {"text": data.text, "sentiment": labels[prediction]}

