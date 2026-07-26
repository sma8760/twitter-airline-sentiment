from fastapi import FastAPI, HTTPException

from app.model import sentiment_model
from app.schemas import PredictionRequest, PredictionResponse


app = FastAPI(
    title="BERT Airline Sentiment API",
    description="Sentiment classification using a fine-tuned BERT model.",
    version="1.0.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "BERT Airline Sentiment API",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "model": "loaded",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        result = sentiment_model.predict(request.text)
        return PredictionResponse(**result)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Prediction error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from error