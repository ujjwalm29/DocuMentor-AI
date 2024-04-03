from openai import OpenAI
from dotenv import load_dotenv
import os
from generation.chat import Chat

load_dotenv()


class ChatOpenAI(Chat):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_message(self, query, context, model: str = "gpt-3.5-turbo"):
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{super().get_system_prompt()}"},
                {"role": "user", "content": f"{super().get_user_rag_prompt(query, context)}"}
            ]
        )

        return completion.choices[0].message.content
