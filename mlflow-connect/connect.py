import os
import mlflow

mlflow.set_tracking_uri(os.getenv("http://localhost:7004"))

mlflow.set_experiment(os.getenv("BERT_Sentiment_Experiment"))

