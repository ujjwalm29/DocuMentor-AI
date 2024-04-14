from llama_parse import LlamaParse, ResultType
from dotenv import load_dotenv
from ingestion.splitters.naive_text_splitter import NaiveTextSplitter
from ingestion.splitters.text_splitter import TextSplitter
import pandas as pd

import os
import logging

load_dotenv()
logger = logging.getLogger(__name__)



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PdfParser:

    def __init__(self, result_type: str = 'md', text_splitter: TextSplitter = NaiveTextSplitter(), parsing_instructs: str = ""):
        self.text_splitter = text_splitter
        self.result_type = result_type
        self.parsing_instructs = parsing_instructs

    def get_text_from_pdf(self, file_path: str):
        logger.debug(f"Starting LlamaParse job.. Result Type :${self.result_type}")
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_PDF_API_KEY"),
            parsing_instruction=self.parsing_instructs
        )
        if self.result_type.lower() == 'txt':
            parser.result_type = ResultType.TXT
        elif self.result_type.lower() == 'md':
            parser.result_type = ResultType.MD
        else:
            raise NotImplementedError

        new_file_name = ''.join(file_path.split('/')[-1].split('.')[:-1]) + '.' + self.result_type.lower()

        # check if PROJECT_ROOT/markdowns/{file}.{extension} exists
        parsed_file_path = os.path.join(PROJECT_ROOT, 'data', 'markdowns', new_file_name)

        if not os.path.exists(parsed_file_path):
            documents = parser.load_data(file_path)

            with open(parsed_file_path, "w") as file:
                file.write(documents[0].text)

            return documents[0].text

        else:
            with open(parsed_file_path, "r") as file:
                document = file.read()

            return document

    def parse_pdf(self):
        document = self.get_text_from_pdf()
        sentences = self.text_splitter.split_text(document)
        return pd.DataFrame(sentences, columns=['sentence'])
