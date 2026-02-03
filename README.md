
# 🧠 BERT Sentiment Analysis API (MLOps Project)

A production-style **Sentiment Analysis system** using BERT, fine-tuned on customer feedback and deployed via FastAPI.

This project demonstrates an **end-to-end MLOps workflow**:
Data → Training → Model Saving → API Serving


---

## ⚙️ Step 1 — Clone the Project

git clone https://github.com/your-username/sentiment-mlops.git  
cd sentiment-mlops

---

## 🐍 Step 2 — Create Virtual Environment

python -m venv venv  

Mac/Linux:
source venv/bin/activate  

Windows:
venv\Scripts\activate  

---

## 📦 Step 3 — Install Dependencies

pip install -r requirements.txt

---

## 🧾 Step 4 — Prepare Dataset

Ensure dataset.csv looks like this:

text,label  
"The cleaner did an amazing job!",positive  
"Very bad service, not satisfied",negative  
"Okay experience",neutral  

Labels:
- positive  
- neutral  
- negative  

---

## 🏋️ Step 5 — Train the BERT Model

python train.py

This will:
- Download pre-trained BERT
- Fine-tune on your dataset
- Save model in /model

---

## 🚀 Step 6 — Run the API Server

uvicorn api:app --reload

API runs at:
http://127.0.0.1:8000  

Swagger Docs:
http://127.0.0.1:8000/docs  

---

## 🔍 Step 7 — Test Prediction

Request:

POST /predict
{
  "text": "The cleaning service was fantastic!"
}

Response:

{
  "text": "The cleaning service was fantastic!",
  "sentiment": "positive"
}

---

## 🧪 Step 8 — Model Retraining

When new data is available:

python train.py

---

## 📊 MLOps Features Demonstrated

✔ Model Training Pipeline  
✔ Model Artifact Saving  
✔ API-based Model Serving  
✔ Ready for Dockerization  
✔ Easy CI/CD integration  
✔ Expandable for Monitoring & Drift Detection  

---

## 🔮 Future Improvements

- Add MLflow model tracking  
- Docker containerization  
- CI/CD pipeline  
- Model performance monitoring  
- Auto-retraining workflows  

---

## 🧑‍💻 Author

Built as an MLOps portfolio project demonstrating production ML deployment.
