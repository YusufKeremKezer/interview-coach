from .settings import settings
from .domain.models import QuestionAnswerPair, EvaluationResponse, UserResponse
from .interview import create_question, evaluate_answer, create_followup_question

global number

async def run_step():
    try:
        match number:
            case 1:
                return await create_question(settings.COMPANY_NAME, settings.ROLE, settings.DIFFICULTY, True)
            case 2:
                return await evaluate_answer(QuestionAnswerPair.question, UserResponse.response, QuestionAnswerPair.answer, True)
            case 3:
                return await create_followup_question(QuestionAnswerPair.question, UserResponse.response, settings.DIFFICULTY, settings.COMPANY_NAME, settings.ROLE, True)
            case 4:
                return await evaluate_answer(QuestionAnswerPair.question, UserResponse.response, QuestionAnswerPair.answer, True)
    finally:
        number += 1