import requests

url = "http://127.0.0.1:8000/predict/"  # changed port to 8000

data = {
    "headline": "Stock markets soar as tech companies report strong earnings"
}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raise error for bad responses

    result = response.json()
    print("✅ Prediction Result:")
    print(f"Headline: {result['headline']}")
    print(f"Predicted Category: {result['predicted_category']}")
    print(f"Confidence: {result['confidence']:.4f}")

except requests.exceptions.RequestException as e:
    print(f"❌ Request failed: {e}")
