from dotenv import load_dotenv
import os
import anthropic
from generation.chat import Chat

load_dotenv()


class ChatClaude(Chat):

    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    def get_message(self, query, context, model: str = "claude-3-haiku-20240307"):
        completion = self.client.messages.create(
            model=model,
            max_tokens=250,
            system=super().get_system_prompt(),
            messages=[
                {"role": "user", "content": f"{super().get_user_rag_prompt(query, context)}"}
            ]
        )

        return completion.content[0].text
