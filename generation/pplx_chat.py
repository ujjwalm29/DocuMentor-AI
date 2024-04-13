from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
from generation.chat import Chat

load_dotenv()
logger = logging.getLogger(__name__)

class ChatPplx(Chat):

    client = OpenAI(api_key=os.getenv("PPLX_API_KEY"), base_url="https://api.perplexity.ai")

    def get_message(self, query, context, model: str = "sonar-small-chat"):
        logger.debug(f"Perplexity API being called query:{query} , context:{context}, model:{model}")

        content = super().get_user_rag_prompt(query, context)
        logger.debug(f"User message for API f{content}")

        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{super().get_system_prompt()}"},
                {"role": "user", "content": f"{content}"}
            ]
        )
        logger.debug(f"Perplexity API Response {completion}")
        return completion.choices[0].message.content
