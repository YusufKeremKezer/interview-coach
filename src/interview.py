from crewai import Agent, Task , Crew, Process
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
import asyncio
from crewai.llm import LLM

llm = LLM(model="gemini/gemini-2.5-flash",base_url="https://generativelanguage.googleapis.com", temperature=0.7)

class QuestionAnswerPair(BaseModel):
    """Model representing a question and its corresponding answer."""
    question: str = Field(..., description="The interview question")
    answer: str = Field(..., description="The interview answer")


search_tool = SerperDevTool()



company_researcher_agent = Agent(
    role="Company Research Specialist",
    goal=" Gather information about the company and create interview questions with answers",
    backstory="""You are an expert in researching companies and  creating technical interview questions. Questions that test both theorical knowledge and technical skills.""",
    tools=[search_tool],
    verbose=True,
    llm=llm
)


question_preparer_agent = Agent(
    role="Question Preparer",
    goal="Prepare interview questions and answers based on the research provided by the Company Research Specialist.",
    backstory="""You are an expert in researching companies and creating technical interview questions.
    You have deep knowledge of tech industry hiring practices and can create relevant
    questions that test both theoretical knowledge and practical skills.""",
    verbose=True,
    llm=llm
    )

follow_up_questioner_agent = Agent(
    role="Follow-up Questioner",
    goal="Create relevant follow-up questions based on the context",
    backstory="""You are an expert technical interviewer who knows how to create
    meaningful follow-up questions that probe deeper into a candidate's knowledge
    and understanding. You can create questions that build upon previous answers
    and test different aspects of the candidate's technical expertise.""",
    verbose=True,
    llm=llm
    )

answer_evaluator_agent = Agent(
    role="Answer Evaluator",
    goal="Evaluate if the given answer is correct for the question",
    backstory="""You are a senior technical interviewer who evaluates answers
    against the expected solution. You know how to identify if an answer is
    technically correct and complete.""",
    verbose=True,
    llm=llm
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

def create_question_preparation_task(difficulty: str) -> Task:
    return Task(
        name="Question Preparation Task",
        description=f"Prepare interview questions and answers of {difficulty} difficulty based on the research provided.",
        agent=question_preparer_agent,
        expected_output="Interview question and its correct answer",
        output_pydantic=QuestionAnswerPair,
    )

def create_evaluation_task(
    question: str, user_answer: str, correct_answer: str
) -> Task:
    return Task(
        description=f"""Evaluate if the given answer is correct for the question:
        Question: {question}
        Answer: {user_answer}
        Correct Answer: {correct_answer}
        Provide:
        1. Whether the answer is correct (Yes/No)
        2. Key points that were correct or missing
        3. A brief explanation of why the answer is correct or incorrect""",
        expected_output="Evaluation of whether the answer is correct for the question with feedback",
        agent=answer_evaluator_agent,
    )

def create_follow_up_question_task(question: str, difficulty: str, company: str, role: str) -> Task:
    return Task(
        description=""" Create a Follow-up Question and its answer based on the provided context below:
        Question: {question}
        Difficulty: {difficulty}
        Company: {company}
        Role: {role}
        """,
        output_pydantic=QuestionAnswerPair,
        expected_output="A follow-up interview question and its correct answer",
        agent=follow_up_questioner_agent,
    )


def create_researcher_crew(company_name: str, role: str, difficulty: str,) -> Crew:
    return Crew(
        agents=[
            company_researcher_agent,
            question_preparer_agent,
        ],
        tasks=[
            create_company_research_task(company_name, role, difficulty),
            create_question_preparation_task(difficulty)],
        name="Researcher Crew",
        description="This crew is responsible for researching companies and preparing interview questions.",
        process=Process.sequential,  # chain process
    )


def create_evaluator_crew(question: str, user_answer: str, correct_answer: str) -> Crew:
    evaluation_crew = Crew(
        agents=[answer_evaluator_agent],
        tasks=[
            create_evaluation_task(
                question=question,
                user_answer=user_answer,
                correct_answer=correct_answer,
            )
        ],
    )
    return evaluation_crew


async def create_followup_crew(question: str, difficulty: str, company: str, role: str) -> Crew:
    followup_crew = Crew(
        agents=[follow_up_questioner_agent],
        tasks=[
            create_follow_up_question_task(
                question=question,
                difficulty=difficulty,
                company=company,
                role=role,
            )
        ],
    )
    return followup_crew


async def run_interview_process(company_name: str, role: str, difficulty: str):
    # Step 1: Research and prepare questions
    researcher_crew = create_researcher_crew(company_name, role, difficulty)
    research_results = researcher_crew.kickoff()

    print(f"Research Results: {research_results}")
    user_answer = input("Enter your answer: ")

    # Step 2: Evaluate user's answer
    evaluator_crew = create_evaluator_crew(research_results.pydantic.question, user_answer, research_results.pydantic.answer)
    evaluation_results = evaluator_crew.kickoff()

    # Step 3: Create follow-up question
    followup_crew = await create_followup_crew(research_results.pydantic.question
, difficulty, company_name, role)

    followup_results = await followup_crew.kickoff()

    return {
        "research": research_results,
        "evaluation": evaluation_results,
        "follow_up": followup_results,
    }


if __name__ == "__main__":
    asyncio.run(run_interview_process(company_name="Google", role="AI Engineer", difficulty="Medium"))