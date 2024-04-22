from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
from generation.chat import Chat
from util import time_function

load_dotenv()
logger = logging.getLogger(__name__)


class ChatOpenAI(Chat):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_final_generated_message(self, query, context, model: str = "gpt-3.5-turbo"):
        content = super().get_user_rag_prompt(query, context)
        return self.call_api(query, model, super().get_system_prompt(), content)

    def get_multiple_queries(self, query, number_of_queries: int = 3):
        queries = self.call_api(query, user_message=super().get_multiple_queries_prompt(query))
        return queries.split('\n')

    @time_function
    def call_api(self, query, model: str = "gpt-3.5-turbo", system_prompt: str = "", user_message=""):
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
