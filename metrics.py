import os
import pickle
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import evaluate

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = "models/transformer_headline/"
DATA_PATH = "data/headlines_cleaned.json"

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_json(DATA_PATH, lines=True)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

df["label"] = le.transform(df["category"])

# Train-test split
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
test_ds = Dataset.from_pandas(test_df[["headline", "label"]])

# -----------------------------
# Load tokenizer + model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

def tokenize(batch):
    return tokenizer(
        batch["headline"],
        truncation=True,
        padding="max_length",
        max_length=60
    )

test_ds = test_ds.map(tokenize, batched=True)
test_ds = test_ds.rename_column("label", "labels")
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------------------
# Metrics
# -----------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1_weighted": f1.compute(predictions=preds, references=p.label_ids, average="weighted")["f1"],
        "f1_macro": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"],
        "precision": precision.compute(predictions=preds, references=p.label_ids, average="weighted")["precision"],
        "recall": recall.compute(predictions=preds, references=p.label_ids, average="weighted")["recall"],
    }

# -----------------------------
# Evaluate
# -----------------------------
trainer = Trainer(model=model, compute_metrics=compute_metrics)

print("🚀 Running Evaluation...")
results = trainer.evaluate(test_ds)

print("\n📊 Final Evaluation Results:")
for metric, value in results.items():
    if isinstance(value, (float, np.floating)):
        print(f"{metric:12s}: {value:.4f}")
    else:
        print(f"{metric:12s}: {value}")
