import json
from pathlib import Path
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# ============================================================
# Configuration
# ============================================================

# Model directory relative to current script
MODEL_PATH = Path("complaintNLP_model")
LABEL_MAP_FILE = MODEL_PATH / "label_map.json"

TEST_COMPLAINTS = [
    ("My internet has been down for 3 days, I work from home", "High"),
    ("There is a gas leak in my building, this is an emergency", "High"),
    ("I have been waiting for a refund for 2 weeks", "Medium"),
    ("The app crashes sometimes when I open it", "Medium"),
    ("I want to update my billing address", "Low"),
    ("Can I get a copy of last month's invoice please", "Low"),
    ("the portal is down and i cant submit my assignment deadline is today", "High"),
    ("a student was attacked in the hostel and security did nothing", "High"),
    ("teacher hasnt returned our assignments for over a week", "Medium"),
    ("the library has been closed for 4 days without any notice", "Medium"),
    ("the cafeteria menu should have more variety", "Low"),
    ("library should stay open later during exams", "Low"),
]


# ============================================================
# Prediction Functions
# ============================================================

def predict_priority(complaint_text, model, tokenizer, label_map, device):
    """Takes a complaint and returns predicted priority label and confidence score."""
    inputs = tokenizer(
        complaint_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_index = str(torch.argmax(outputs.logits).item())
    priority = label_map[predicted_index]

    # Get confidence score
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence = round(torch.max(probabilities).item() * 100, 2)

    return priority, confidence


# ============================================================
# Main Execution
# ============================================================

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model folder not found at '{MODEL_PATH.resolve()}'. "
            "Please ensure training has run or model files are placed in this folder."
        )

    if not LABEL_MAP_FILE.exists():
        raise FileNotFoundError(f"Label map not found at '{LABEL_MAP_FILE.resolve()}'.")

    # Load label map
    with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")

    print("=" * 65)
    print("             COMPLAINT PRIORITY PREDICTION RESULTS")
    print("=" * 65)

    correct = 0
    for complaint, expected in TEST_COMPLAINTS:
        predicted, confidence = predict_priority(
            complaint, model, tokenizer, label_map, device
        )
        status = "✅ Correct" if predicted == expected else "❌ Wrong"
        if predicted == expected:
            correct += 1

        print(f"\nComplaint  : {complaint}")
        print(f"Expected   : {expected}")
        print(f"Predicted  : {predicted}  (Confidence: {confidence}%)")
        print(f"Status     : {status}")
        print("-" * 65)

    accuracy = round((correct / len(TEST_COMPLAINTS)) * 100, 2)
    print(f"\n📊 Quick Accuracy: {correct}/{len(TEST_COMPLAINTS)} correct = {accuracy}%")

    # Probability test on a single custom string
    test_text = "There is a gas leak in girls hostel!"
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    print("\nProbability for each class:")
    for idx, prob in enumerate(probs):
        print(f"  {label_map[str(idx)]}: {round(prob.item() * 100, 2)}%")


if __name__ == "__main__":
    main()