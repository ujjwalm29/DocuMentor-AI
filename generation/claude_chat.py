from dotenv import load_dotenv
import os
import anthropic
import logging
from generation.chat import Chat

load_dotenv()
logger = logging.getLogger(__name__)


class ChatClaude(Chat):

    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    def get_message(self, query, context, model: str = "claude-3-haiku-20240307"):
        logger.debug(f"Claude API being called query:{query} , context:{context}, model:{model}")

        content = super().get_user_rag_prompt(query, context)
        logger.debug(f"User message for API f{content}")

        completion = self.client.messages.create(
            model=model,
            max_tokens=250,
            system=super().get_system_prompt(),
            messages=[
                {"role": "user", "content": f"{content}"}
            ]
        )
        logger.debug(f"Claude API Response {completion}")
        return completion.content[0].text
