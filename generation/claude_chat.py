from dotenv import load_dotenv
import os
import anthropic
import logging
from generation.chat import Chat

load_dotenv()
logger = logging.getLogger(__name__)


class ChatClaude(Chat):

    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    def get_final_generated_message(self, query, context, model: str = "claude-3-haiku-20240307"):
        content = super().get_user_rag_prompt(query, context)
        return self.call_api(query, model, super().get_system_prompt(), content)

    def get_multiple_queries(self, query):
        queries = self.call_api(query, user_message=super().get_multiple_queries_prompt(query))
        return queries.split('\n')


    def call_api(self, query, model: str = "claude-3-haiku-20240307", system_prompt: str = "", user_message=""):
        logger.debug(f"Claude API being called query:{query} , context:{user_message}, model:{model}")

        if system_prompt == "":
            system_prompt = "You are a helpful AI assistant."

        completion = self.client.messages.create(
            model=model,
            max_tokens=250,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"{user_message}"}
            ]
        )
        logger.debug(f"Claude API Response {completion}")
        return completion.content[0].text
