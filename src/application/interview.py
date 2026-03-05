from crewai import Agent, Task , Crew, Process, LLM
from crewai_tools import SerperDevTool
from ..domain.models import QuestionAnswerPair, EvaluationResponse
from ..settings import Settings
import logging
settings = Settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=settings.OPENROUTER_API_KEY,
)

search_tool = SerperDevTool()

company_researcher_agent = Agent(
    role="Company Research Specialist Voice Agent",
    goal=" Gather information about the company and create interview questions and answers",
    backstory="""You are an expert in researching companies and  creating technical interview questions. Questions that test both theorical knowledge and technical skills.""",
    tools=[search_tool],
    verbose=True,
    llm=llm,
    memory=False
)

speech_router_agent = Agent(
    role="Speech Router Agent",
    goal="Message effectiveness evaluation",
    backstory=""" 
    You are senior speech expert
    You are dealing with barge-in or complete responses.
    You are a router of interview responses.
    """,
    llm=llm,
    verbose=True,
)

speech_routing_task = Task(
    description="""Route the given message to appropriate outcome: {message}

    If message is complete return "Complete"
    If message is interrupted return "Barge-in"

    """,
    agent=speech_router_agent,
    expected_output="Complete or Barge-in",
)

question_preparer_agent = Agent(
    role="Question Preparer Voice Agent",
    goal="Prepare interview question and answer based on the research provided by the Company Research Specialist.",
    backstory="""You are an expert in researching companies and creating technical interview questions.
    You have deep knowledge of tech industry hiring practices and can create relevant
    questions that test both theoretical knowledge and practical skills.""",
    verbose=True,
    llm=llm,
    memory=False
    )

follow_up_questioner_agent = Agent(
    role="Follow-up Questioner Voice Agent",
    goal="Create relevant follow-up question and answer based on the context",
    backstory="""You are an expert technical interviewer who knows how to create
    meaningful follow-up questions that probe deeper into a candidate's knowledge
    and understanding. You can create questions that build upon previous answers
    and test different aspects of the candidate's technical expertise.""",
    verbose=True,
    llm=llm,
    )

answer_evaluator_agent = Agent(
    role="Answer Evaluator Voice Agent",
    goal="Evaluate if the given answer is correct for the question and give feedback to the user",
    backstory="""You are a senior technical interviewer who evaluates answers
    against the expected solution. You know how to identify if an answer is
    technically correct and complete.""",
    verbose=True,
    llm=llm,
)

router_agent = Agent(
    role="Router Agent",
    goal="Route the given user response to the appropriate option",
    backstory="""You are a router agent that checks the user answer if it is complete or its asking for further questions to clarification""",
    verbose=True,
    llm=llm,
)

async def create_router_task() -> Task:
    return Task(
        description="""Route the given user response to the appropriate option: {user_response}
        
        Options:
        1. "Evaluate Answer"
        2. Create Follow-up Question
        3. End Interview
        ",
        agent=router_agent,
        expected_output="The appropriate option to route the user response 
        """,
    )

async def create_company_research_task() -> Task:
    """Create a task for researching a company."""
    return Task(
        name="Company Research Task",
        description="""Use the search tool and Research the {company} company and gather information about:
        1. Their technical interview process
        2. Common interview questions for {role} positions at {difficulty} difficulty level
        3. Technical stack and requirements
        
        Provide a summary of your findings.""",
        agent=company_researcher_agent,
        expected_output="A report about the company's technical requirements and interview process",
    )

async def create_question_preparation_task(context: Task) -> Task:
    return Task(
        name="Question Preparation Task",
        description="Prepare interview question and answer of {difficulty} difficulty based on the research provided.",
        agent=question_preparer_agent,
        expected_output="An interview question and its correct answer",
        output_pydantic=QuestionAnswerPair,
        context=[context],
    )

async def create_evaluation_task(
) -> Task:
    return Task(
        description="""Evaluate if the given answer is correct for the question:
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

async def create_follow_up_question_task() -> Task:
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


async def create_researcher_crew() -> Crew:
    research_task = await create_company_research_task()
    question_task = await create_question_preparation_task(research_task)
    return Crew(
        agents=[
            company_researcher_agent,
            question_preparer_agent,
        ],
        tasks=[
            research_task,
            question_task,
        ],
        memory=False,
        name="Researcher Crew",
        verbose=True,
        description="This crew is responsible for researching companies and preparing interview questions.",
        process=Process.sequential,  # chain process
    )


async def create_evaluator_crew() -> Crew:
    evaluation_task = await create_evaluation_task()
    
    evaluation_crew = Crew(
        agents=[answer_evaluator_agent],
        tasks=[
            evaluation_task,
        ],
        memory=False,
    )
    return evaluation_crew


async def create_followup_crew() -> Crew:
    follow_up_task = await create_follow_up_question_task()
    followup_crew = Crew(
        agents=[follow_up_questioner_agent],
        tasks=[follow_up_task],
        memory=False,
    )
    return followup_crew




