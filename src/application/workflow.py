
from ..domain.models import QuestionAnswerPair
import asyncio
from ..domain.models import InterviewState, stt_message
from crewai.flow.flow import Flow, router, start, listen
from .interview import create_researcher_crew, create_evaluator_crew
from .text_to_speech import tts

class InterviewWorkflow(Flow[InterviewState]):

    @start()
    async def create_question(self):    # Step 1: Research and prepare questions
        
        researcher_crew = await create_researcher_crew(self.state.company_name, self.state.role, self.state.difficulty)
        research_results = await researcher_crew.kickoff_async()

        await tts(research_results.pydantic.question)
        print(f"Soru: {research_results.pydantic.question}")

        return QuestionAnswerPair(
            question=research_results.pydantic.question,
            answer=research_results.pydantic.answer,
        )

    @listen("create_question")
    async def wait_for_user_answer(self, qna_pair):
        print(f"Soru soruldu: {qna_pair.question}")
        print("Mikrofon dinleniyor... Cevabınızı söyleyin.")

        # Burada STT'den cevap bekle (queue'dan al)
        user_response = await self._wait_for_stt_answer()

        # Cevabı state'e kaydet
        self.state.user_response = user_response
        print(f"Cevap: {user_response}")
        return user_response

    async def _wait_for_stt_answer(self) -> str:
        while True:
            if stt_message.message != "":
                message = stt_message.message
                stt_message.message = "" # To clear the data for next time
                return message

            await asyncio.sleep(0.3) 


    @listen(wait_for_user_answer)
    async def evaluate_answer(self, user_response):
        evaluator_crew = await create_evaluator_crew(self.state.question, user_response, self.state.answer)
        evaluation_results = await evaluator_crew.kickoff_async()
        self.state.evaluation_result = evaluation_results.pydantic.evaluation_result
        print(f"Evaluation Result: {self.state.evaluation_result}")
        await tts(self.state.evaluation_result)

        return self.state.evaluation_result
    
if __name__ == "__main__":
    interview_workflow = InterviewWorkflow()
    asyncio.run(interview_workflow.kickoff_async())

