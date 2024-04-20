from groq import Groq
from dotenv import load_dotenv
import os
import logging
from generation.chat import Chat

load_dotenv()
logger = logging.getLogger(__name__)


class ChatGroq(Chat):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def get_final_generated_message(self, query, context, model: str = "llama3-8b-8192"):
        content = super().get_user_rag_prompt(query, context)
        return self.call_api(query, model, super().get_system_prompt(), content)

    def get_multiple_queries(self, query):
        queries = self.call_api(query, user_message=super().get_multiple_queries_prompt(query))
        return queries.split('\n')

    def call_api(self, query, model: str = "llama3-8b-8192", system_prompt: str = "", user_message=""):
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
