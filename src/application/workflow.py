
from ..domain.models import QuestionAnswerPair
import asyncio
from ..domain.models import InterviewState, stt_message
from crewai.flow.flow import Flow, router, start, listen
from .interview import create_researcher_crew, create_evaluator_crew, create_followup_crew
from .text_to_speech import tts
from .speech_to_text import SttClient




class InterviewWorkflow(Flow[InterviewState]):
    def __init__(self):
        super().__init__()
        self.researcher_crew = None
        self.evaluator_crew = None
        self.followup_crew = None
        self.stt = None
        self.tts = None

    @classmethod
    async def create(cls) -> "InterviewWorkflow":
        instance = cls()
        instance.researcher_crew = await create_researcher_crew()
        instance.evaluator_crew = await create_evaluator_crew()
        instance.followup_crew = await create_followup_crew()
        instance.stt = SttClient()
        instance.tts = tts
        return instance

    @start()
    async def start_interview(self):
        self.stt.start_stt()
        
    @listen("start_interview")
    @listen("evaluate_follow_up_answer")
    async def prepare_question(self):    # Step 1: Research and prepare questions
        
        research_results = await self.researcher_crew.kickoff_async(
            {
                "company": self.state.company,
                "role": self.state.role,
                "difficulty": self.state.difficulty,
            }
        )
        self.state.question = research_results.pydantic.question
        self.state.answer = research_results.pydantic.answer

        await self.tts(self.state.question)
        print(f"Soru: {self.state.question}")

   
    @listen("prepare_question")
    async def get_user_response(self):

        user_response = await self._wait_for_speech()
        self.state.user_response = user_response
 
    async def _wait_for_speech(self) -> str:
        self.stt.mic_enabled.set()
        while True:
            if self.stt.message != "":
                self.stt.mic_enabled.clear()
                return self.stt.message
            await asyncio.sleep(0.3) 
    """   
        @router("get_user_response")
        async def route_user_response(self, user_response):
            eval_response = await self.evaluator_crew.kickoff_async({
                "question": self.state.question,
                "user_answer": self.state.user_response,
                "correct_answer": self.state.answer,
            })
            if eval_response == "Complete":
                return "evaluate_answer"
            elif eval_response == "Barge-in":
                return "prepare_question"
            else:
                return "error" 
    """
    @listen("get_user_response")
    async def evaluate_answer(self):
        evaluation_results = await self.evaluator_crew.kickoff_async(
            {
                "question": self.state.question,
                "user_answer": self.state.user_response,
                "correct_answer": self.state.answer,
            }
        )
        self.state.evaluation_result = evaluation_results.pydantic.evaluation_result
        print(f"Evaluation Result: {self.state.evaluation_result}")
        await tts(self.state.evaluation_result)

 
    @listen("evaluate_answer")
    async def follow_up_question(self):
        follow_up_question = await self.followup_crew.kickoff_async({
            "question": self.state.question,
            "user_answer": self.state.user_response,
            "difficulty": self.state.difficulty,
            "company": self.state.company,
            "role": self.state.role,
        })

        await self.tts(follow_up_question.pydantic.question)

    @listen("follow_up_question")
    async def get_follow_up_user_response(self):
        user_response = await self._wait_for_speech()
        self.state.user_response = user_response
        
    @listen("get_follow_up_user_response")
    async def evaluate_follow_up_answer(self):
        evaluation_results = await self.evaluator_crew.kickoff_async({
            "question": self.state.question,
            "user_answer": self.state.user_response,
            "correct_answer": self.state.answer,
        })
        self.state.evaluation_result = evaluation_results.pydantic.evaluation_result
        print(f"Evaluation Result: {self.state.evaluation_result}")
        await tts(self.state.evaluation_result)
    
async def main():
    interview_workflow = await InterviewWorkflow.create()
    await interview_workflow.kickoff_async()
 

    

if __name__ == "__main__":
    asyncio.run(main())
 

