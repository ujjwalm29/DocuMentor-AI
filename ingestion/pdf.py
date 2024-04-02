from llama_parse import LlamaParse, ResultType
from dotenv import load_dotenv

import os

load_dotenv()

parser = LlamaParse(
    api_key=os.getenv("LLAMA_PDF_API_KEY"),  # can also be set in your env as LLAMA_CLOUD_API_KEY
)


def parse_pdf(file_name: str, result_type: str = 'md'):
    if result_type.lower() == 'txt':
        parser.result_type = ResultType.TXT
    elif result_type.lower() == 'md':
        parser.result_type = ResultType.MD
    else:
        raise NotImplementedError

    documents = parser.load_data(f"{file_name}")

    return documents
