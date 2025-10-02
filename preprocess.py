# preprocess.py
import re
import nltk
from nltk.corpus import stopwords
import unicodedata
import pandas as pd

# -------------------------------
# Download English stopwords (once)
# -------------------------------
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# -------------------------------
# Text cleaning function
# -------------------------------
def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Clean input text by:
    - Normalizing Unicode (NFKC) for multilingual scripts
    - Lowercasing English letters only
    - Removing URLs, digits, punctuation, and special characters
    - Preserving Unicode letters (Hindi, Tamil, etc.)
    - Optionally removing English stopwords and single-character tokens
    - Handling None/NaN or non-string inputs safely

    Args:
        text (str): Input text to clean
        remove_stopwords (bool): Whether to remove English stopwords

    Returns:
        str: Cleaned text
    """

    # Handle None/NaN/non-string safely
    if text is None or pd.isna(text) or not isinstance(text, str):
        return ""

    # Normalize unicode (avoids duplicate diacritics in Indic scripts)
    text = unicodedata.normalize("NFKC", text).strip()

    # Lowercase only English letters
    text = text.lower()

    # Remove URLs
    text = re.sub(r"(http|https)://\S+|www\.\S+", " ", text)

    # Remove digits
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation/special chars but keep all Unicode letters + spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize by whitespace
    tokens = text.split()

    if remove_stopwords:
        # Remove English stopwords + single-character tokens
        tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 1]

    return " ".join(tokens)


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    sample_text = "Breaking News: AI model achieves 98% accuracy! Visit https://example.com for more info."
    cleaned = clean_text(sample_text)
    print("Original:", sample_text)
    print("Cleaned: ", cleaned)
