from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "artifacts" / "bert_model"


class SentimentModel:
    def __init__(self) -> None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Model directory was not found: {MODEL_DIR}"
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading tokenizer from: {MODEL_DIR}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )

        print(f"Loading model from: {MODEL_DIR}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on: {self.device}")
        print(f"Label mapping: {self.model.config.id2label}")

    def get_label(self, class_id: int) -> str:
        id2label = self.model.config.id2label

        label = id2label.get(class_id)

        if label is None:
            label = id2label.get(str(class_id))

        if label is None:
            label = f"LABEL_{class_id}"

        return str(label)

    def predict(self, text: str) -> dict[str, Any]:
        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("Text cannot be empty.")

        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )[0]

        predicted_id = int(torch.argmax(probabilities).item())
        predicted_label = self.get_label(predicted_id)
        confidence = float(probabilities[predicted_id].item())

        probability_results = []

        for class_id, probability in enumerate(probabilities):
            probability_results.append(
                {
                    "label": self.get_label(class_id),
                    "probability": round(
                        float(probability.item()),
                        6,
                    ),
                }
            )

        return {
            "text": cleaned_text,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 6),
            "probabilities": probability_results,
        }


sentiment_model = SentimentModel()