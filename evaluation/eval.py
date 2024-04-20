from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument

from DocumentController import DocumentController
from generation.openai_chat import ChatOpenAI

tru = Tru()

class RAG_from_scratch:
    @instrument
    def retrieve(self, query: str) -> list:
        doc = DocumentController()
        results = doc.search_and_retrieve_result(query)
        return results

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        generate = ChatOpenAI()
        answer = generate.get_message(query, context_str)
        return answer

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query)
        completion = self.generate_completion(query, context_str)
        return completion

rag = RAG_from_scratch()

from trulens_eval import Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI

import numpy as np

provider = OpenAI()

grounded = Groundedness(groundedness_provider=provider)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name = "Answer Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on_output()
)

# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets.collect())
    .aggregate(np.mean)
)

from trulens_eval import TruCustomApp
tru_rag = TruCustomApp(rag,
    app_id = 'RAG v1',
    feedbacks = [f_groundedness, f_answer_relevance, f_context_relevance])


with tru_rag as recording:
    rag.query("Why should we perform network simulation?")

tru.get_leaderboard(app_ids=["RAG v1"])

tru.run_dashboard()
