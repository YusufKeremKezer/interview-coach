from pydantic import BaseModel, Field

class QuestionAnswerPair(BaseModel):
    """Model representing a question and its corresponding answer."""
    question: str = Field(..., description="The interview question")
    answer: str = Field(..., description="The interview answer")


class EvaluationResponse(BaseModel):
    """Model representing a evaluation response."""
    evaluation_result: str = Field(..., description="The evaluation of the user's response")
