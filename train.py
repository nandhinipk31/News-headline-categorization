import os
import pickle
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import shutil

if __name__ == "__main__":
    # -----------------------------
    # Config
    # -----------------------------
    MODEL_NAME = "distilbert-base-multilingual-cased"
    DATA_PATH = "data/merged_news.csv"
    OUTPUT_DIR = "models/distilmbert_headline/"
    MAX_LEN = 64
    BATCH_SIZE = 32          # adjust if CPU memory is limited
    EPOCHS = 1               # fast CPU training
    GRAD_ACCUM = 2           # accumulate gradients to simulate bigger batch

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------
    # Load dataset in chunks
    # -----------------------------
    print("📥 Loading dataset...")
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=20000, usecols=["Title", "Category"]):
        chunk = chunk.dropna(subset=["Title", "Category"])
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # reduce dataset for fast CPU testing
    df = df.head(10000)

    # -----------------------------
    # Encode labels
    # -----------------------------
    le = LabelEncoder()
    df["labels"] = le.fit_transform(df["Category"])
    num_labels = len(le.classes_)
    print(f"📑 Categories: {list(le.classes_)}")

    # -----------------------------
    # HuggingFace dataset
    # -----------------------------
    dataset = Dataset.from_pandas(df[["Title", "labels"]])
    dataset = dataset.rename_column("Title", "text")

    dataset = dataset.train_test_split(test_size=0.02, seed=42)
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    # -----------------------------
    # Tokenizer
    # -----------------------------
    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )

    train_ds = train_ds.map(tokenize, batched=True, num_proc=1)
    test_ds = test_ds.map(tokenize, batched=True, num_proc=1)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # -----------------------------
    # Model
    # -----------------------------
    print("🧠 Building model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # -----------------------------
    # Metrics
    # -----------------------------
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "f1": f1.compute(predictions=preds, references=p.label_ids, average="weighted")["f1"],
            "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"]
        }

    # -----------------------------
    # Training Arguments (CPU-optimized)
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,

        # CPU optimizations
        fp16=False,
        dataloader_num_workers=1,
        evaluation_strategy="no",  # skip evaluation during training
        save_strategy="no",        # skip checkpoints
        load_best_model_at_end=False,
        report_to="none",
        overwrite_output_dir=True
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("🚀 Training...")
    trainer.train()

    # -----------------------------
    # Evaluate
    # -----------------------------
    print("📊 Evaluating on test set...")
    metrics = trainer.evaluate(test_ds)
    print(f"✅ Test Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"✅ Test F1 Score: {metrics['eval_f1']:.4f}")

    # -----------------------------
    # Save model & tokenizer
    # -----------------------------
    print("💾 Saving model folder...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # -----------------------------
    # Try compressing model
    # -----------------------------
    zip_path = OUTPUT_DIR.rstrip("/") + ".zip"
    print(f"📦 Compressing model to {zip_path} ...")
    try:
        shutil.make_archive(base_name=OUTPUT_DIR.rstrip("/"), format='zip', root_dir=OUTPUT_DIR)
        print("✅ Model saved and compressed!")
    except OSError as e:
        print(f"⚠️ Skipping compression due to error: {e}")
        print("✅ Model folder is still saved at:", OUTPUT_DIR)
