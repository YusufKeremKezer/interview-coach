from crewai import Agent, Task , Crew, Process, LLM
from crewai_tools import SerperDevTool

from .settings import Settings
import logging
from .domain.models import QuestionAnswerPair, EvaluationResponse
from .text_to_speech import message_to_speech

settings = Settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = LLM(
    model="openrouter/tngtech/deepseek-r1t2-chimera:free",
    api_key=settings.OPENROUTER_API_KEY,
)

search_tool = SerperDevTool()




company_researcher_agent = Agent(
    role="Company Research Specialist",
    goal=" Gather information about the company and create interview questions and answers",
    backstory="""You are an expert in researching companies and  creating technical interview questions. Questions that test both theorical knowledge and technical skills.""",
    tools=[search_tool],
    verbose=True,
    llm=llm,
)


question_preparer_agent = Agent(
    role="Question Preparer",
    goal="Prepare interview question and answer based on the research provided by the Company Research Specialist.",
    backstory="""You are an expert in researching companies and creating technical interview questions.
    You have deep knowledge of tech industry hiring practices and can create relevant
    questions that test both theoretical knowledge and practical skills.""",
    verbose=True,
    llm=llm
    )

follow_up_questioner_agent = Agent(
    role="Follow-up Questioner",
    goal="Create relevant follow-up question and answer based on the context",
    backstory="""You are an expert technical interviewer who knows how to create
    meaningful follow-up questions that probe deeper into a candidate's knowledge
    and understanding. You can create questions that build upon previous answers
    and test different aspects of the candidate's technical expertise.""",
    verbose=True,
    llm=llm,
    )

answer_evaluator_agent = Agent(
    role="Answer Evaluator",
    goal="Evaluate if the given answer is correct for the question",
    backstory="""You are a senior technical interviewer who evaluates answers
    against the expected solution. You know how to identify if an answer is
    technically correct and complete.""",
    verbose=True,
    llm=llm,
)

def create_company_research_task(company_name: str, role: str, difficulty: str) -> Task:
    """Create a task for researching a company."""
    return Task(
        name="Company Research Task",
        description=f"""Research {company_name} and gather information about:
        1. Their technical interview process
        2. Common interview questions for {role} positions at {difficulty} difficulty level
        3. Technical stack and requirements
        
        Provide a summary of your findings.""",
        agent=company_researcher_agent,
        expected_output="A report about the company's technical requirements and interview process",
    )

def create_question_preparation_task(difficulty: str, context: Task) -> Task:
    return Task(
        name="Question Preparation Task",
        description=f"Prepare interview question and answer of {difficulty} difficulty based on the research provided.",
        agent=question_preparer_agent,
        expected_output="An interview question and its correct answer",
        output_pydantic=QuestionAnswerPair,
        context=[context],
    )

def create_evaluation_task(
    question: str, user_answer: str, correct_answer: str
) -> Task:
    return Task(
        description=f"""Evaluate if the given answer is correct for the question:
        Question: {question}
        User Answer: {user_answer}
        Correct Answer: {correct_answer}
        Provide:
        1. Whether the answer is correct (Yes/No)
        2. Key points that were correct or missing
        3. A brief explanation of why the answer is correct or incorrect""",
        expected_output="Evaluation of whether the answer is correct for the question with feedback",
        output_pydantic=EvaluationResponse,
        agent=answer_evaluator_agent,
    )

def create_follow_up_question_task(question: str, user_answer: str, difficulty: str, company: str, role: str) -> Task:
    return Task(
        description=""" Create a Follow-up Question and its answer based on the provided context below:
        Question: {question}
        User Answer: {user_answer}
        Difficulty: {difficulty}
        Company: {company}
        Role: {role}

        """,
        output_pydantic=QuestionAnswerPair,
        expected_output="A follow-up interview question and its correct answer",
        agent=follow_up_questioner_agent,
    )


def create_researcher_crew(company_name: str, role: str, difficulty: str,) -> Crew:
    research_task = create_company_research_task(company_name, role, difficulty)
    question_task = create_question_preparation_task(difficulty, research_task)
    return Crew(
        agents=[
            company_researcher_agent,
            question_preparer_agent,
        ],
        tasks=[
            research_task,
            question_task,
        ],
        name="Researcher Crew",
        description="This crew is responsible for researching companies and preparing interview questions.",
        process=Process.sequential,  # chain process
    )


def create_evaluator_crew(question: str, user_answer: str, correct_answer: str) -> Crew:
    global evaluation_task
    evaluation_task = create_evaluation_task(question, user_answer, correct_answer)
    
    evaluation_crew = Crew(
        agents=[answer_evaluator_agent],
        tasks=[
            evaluation_task,
        ]
    )
    return evaluation_crew


def create_followup_crew(question: str, user_answer: str, difficulty: str, company: str, role: str) -> Crew:
    followup_crew = Crew(
        agents=[follow_up_questioner_agent],
        tasks=[create_follow_up_question_task(
                question=question,
                user_answer=user_answer,
                difficulty=difficulty,
                company=company,
                output_pydantic=QuestionAnswerPair,
                role=role,
                context = [evaluation_task]
            )]
        
    )
    return followup_crew


async def create_question(company_name: str, role: str, difficulty: str, is_voice_chat:bool = False):
        # Step 1: Research and prepare questions
    researcher_crew = create_researcher_crew(company_name, role, difficulty)
    research_results = researcher_crew.kickoff()

    # Extract the first question and answer for evaluation
    question = research_results.pydantic.question

    if is_voice_chat:
        await message_to_speech(question)
    else:
        return question
        

async def evaluate_answer(question: str, user_answer: str, correct_answer: str, is_voice_chat: bool = False):
    evaluator_crew = create_evaluator_crew(question, user_answer, correct_answer)
    evaluation_results = evaluator_crew.kickoff()
    if is_voice_chat:
        await message_to_speech(evaluation_results.pydantic.evaluation_result)
    else:
        return evaluation_results.pydantic.evaluation_result


async def create_followup_question(question: str, user_answer: str, difficulty: str, company: str, role: str, is_voice_chat: bool = False):
    followup_crew = create_followup_crew(question, user_answer, difficulty, company, role)
    followup_results = followup_crew.kickoff()
    if is_voice_chat:
        await message_to_speech(followup_results.pydantic.question)
    else:
        return followup_results.pydantic.question
