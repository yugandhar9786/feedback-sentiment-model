import os
import pandas as pd
import torch
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# =====================================================
# 🔗 CONNECT TO MLFLOW TRACKING SERVER
# =====================================================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))

# =====================================================
# 📂 LOAD DATA
# =====================================================
df = pd.read_csv("data/customer_feedback_sentiment.csv")
df["label"] = df["label"].astype(int)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["feedback_text"], df["label"], test_size=0.2, random_state=42
)

# =====================================================
# 🔤 TOKENIZER
# =====================================================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
val_dataset = Dataset.from_dict({"text": val_texts.tolist(), "label": val_labels.tolist()})

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# =====================================================
# 🤖 MODEL
# =====================================================
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    problem_type="single_label_classification"
)

# =====================================================
# 📊 METRICS FUNCTION
# =====================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_score": f1_score(labels, preds),
    }

# =====================================================
# ⚙ TRAINING CONFIG
# =====================================================
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    save_strategy="no",  # no checkpoints needed
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# =====================================================
# 🚀 TRAIN + LOG TO MLFLOW (METRICS ONLY)
# =====================================================
with mlflow.start_run():

    mlflow.log_param("model_name", "bert-base-uncased")
    mlflow.log_param("epochs", 3)
    mlflow.log_param("batch_size", 8)
    mlflow.log_param("learning_rate", 2e-5)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()

    mlflow.log_metric("accuracy", results["eval_accuracy"])
    mlflow.log_metric("f1_score", results["eval_f1_score"])

print("✅ Training complete. Metrics logged to MLflow.")
