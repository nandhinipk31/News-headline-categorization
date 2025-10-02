# utils.py
import unicodedata
import pandas as pd
from transformers import AutoTokenizer
import torch

# Default maximum sequence length (should match training)
MAX_LEN = 64  # matches training script

# -------------------------------
# Unicode normalization
# -------------------------------
def normalize_unicode(text: str) -> str:
    """
    Normalize text to NFKC form (handles mixed Unicode forms in Indian & global languages).
    Converts non-string/NaN values to empty strings.

    Args:
        text (str): Input text

    Returns:
        str: Normalized text
    """
    if text is None or pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFKC", text).strip()


# -------------------------------
# Load Hugging Face tokenizer
# -------------------------------
def load_hf_tokenizer(model_name_or_path: str = "distilbert-base-multilingual-cased", max_len: int = MAX_LEN, verbose: bool = False):
    """
    Load a Hugging Face tokenizer for transformer-based models.

    Args:
        model_name_or_path (str): Model name (HF Hub) or local path.
        max_len (int): Maximum sequence length for padding/truncation.
        verbose (bool): Logs progress if True.

    Returns:
        tokenizer (AutoTokenizer): Loaded Hugging Face tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if verbose:
        print(f"✅ Loaded Hugging Face tokenizer from: {model_name_or_path}")
        print(f"📏 Max sequence length set to: {max_len}")

    return tokenizer


# -------------------------------
# Convert texts to padded token tensors
# -------------------------------
def texts_to_padded(tokenizer, texts, max_len: int = MAX_LEN, verbose: bool = False):
    """
    Convert raw texts into padded tokenized inputs for Hugging Face models.

    Args:
        tokenizer (AutoTokenizer): Hugging Face tokenizer.
        texts (list[str] or pd.Series): Input texts.
        max_len (int): Maximum sequence length after padding.
        verbose (bool): Logs progress if True.

    Returns:
        dict: Dictionary of input_ids and attention_mask for PyTorch models.
    """
    if not texts:
        return {"input_ids": torch.empty(0, max_len, dtype=torch.long),
                "attention_mask": torch.empty(0, max_len, dtype=torch.long)}

    # Normalize all texts safely
    normalized_texts = [normalize_unicode(t) for t in texts]

    if verbose:
        print(f"🔹 Tokenizing {len(normalized_texts)} texts...")

    encoded = tokenizer(
        normalized_texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )

    if verbose:
        print(f"📏 Encoded input_ids shape: {encoded['input_ids'].shape}")
        print(f"📏 Encoded attention_mask shape: {encoded['attention_mask'].shape}")

    return encoded
