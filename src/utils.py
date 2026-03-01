from .settings import settings
from .domain.models import question_answer_pair, UserResponse
from .interview import create_question, evaluate_answer, create_followup_question

number = 1

async def run_step():
    global number
    try:
        match number:
            case 0:
                return await create_question(settings.COMPANY_NAME, settings.ROLE, settings.DIFFICULTY, True)
            case 1:
                return await evaluate_answer(question_answer_pair.question, UserResponse.response, question_answer_pair.answer, True)
            case 2:
                return await create_followup_question(question_answer_pair.question, UserResponse.response, settings.DIFFICULTY, settings.COMPANY_NAME, settings.ROLE, True)
            case 3:
                return await evaluate_answer(question_answer_pair.question, UserResponse.response, question_answer_pair.answer, True)
    finally:
        number += 1
        if number == 4:
            number = 0