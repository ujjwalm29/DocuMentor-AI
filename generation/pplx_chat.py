from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
from generation.chat import Chat

load_dotenv()
logger = logging.getLogger(__name__)

class ChatPplx(Chat):

    client = OpenAI(api_key=os.getenv("PPLX_API_KEY"), base_url="https://api.perplexity.ai")

    def get_final_generated_message(self, query, context, model: str = "sonar-small-chat"):
        content = super().get_user_rag_prompt(query, context)
        return self.call_api(query, model, super().get_system_prompt(), content)

    def get_multiple_queries(self, query, number_of_queries: int = 3):
        queries = self.call_api(query, user_message=super().get_multiple_queries_prompt(query))
        return queries.split('\n')

    def call_api(self, query, model: str = "sonar-small-chat", system_prompt: str = "", user_message=""):
        logger.debug(f"OpenAI API being called query:{query} , user_message:{user_message}, model:{model}")

        if system_prompt == "":
            system_prompt = "You are a helpful AI assistant."

        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{user_message}"}
            ]
        )
        logger.debug(f"OpenAI API Response {completion}")
        return completion.choices[0].message.content
