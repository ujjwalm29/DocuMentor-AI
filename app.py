from DocumentController import DocumentController
from ingestion.pdf import PdfParser
from ingestion.splitters.recursive_splitter import RecursiveTextSplitter
from generation.openai_chat import ChatOpenAI
from util import setup_logging

import logging
import os
from dotenv import load_dotenv

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

logger.info("Starting...")
file_path = 'data/pdf/amazon-dynamo-sosp2007.pdf'
pkl_file_name = ''.join(file_path.split('/')[-1].split('.')[:-1]) + '.pkl'
pkl_file_path = os.path.join('data', 'pkl', pkl_file_name)


query = "What is N, R and W?"

controller = DocumentController()
controller.process_text_and_store(file_path=file_path)
context = controller.search_and_retrieve_result(query)

generate = ChatOpenAI()
print(generate.get_message(query, context))
