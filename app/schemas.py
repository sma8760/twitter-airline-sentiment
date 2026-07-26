from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Text whose sentiment should be predicted.",
    )


class ClassProbability(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    probabilities: list[ClassProbability]