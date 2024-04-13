from DocumentController import DocumentController
from ingestion.pdf import PdfParser
from ingestion.splitters.recursive_splitter import RecursiveTextSplitter
from embeddings.LocalHFEmbeddings import LocalEmbeddings
from generation.openai_chat import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()

file_path = 'data/pdf/amazon-dynamo-sosp2007.pdf'
pkl_file_name = ''.join(file_path.split('/')[-1].split('.')[:-1]) + '.pkl'
pkl_file_path = os.path.join('data', 'pkl', pkl_file_name)

embeddings = LocalEmbeddings(embeddings_model="mixedbread-ai/mxbai-embed-large-v1")
pdf_parser = PdfParser(file_path=file_path, parsing_instructs="""
You are parsing a research paper. DO NOT parse or include the references section or any metadata or acknowledgments in the output.
Convert tables into a list of facts. Do not include "Research Paper" in the final md file.
""", text_splitter=RecursiveTextSplitter(chunk_size=300))


query = "What is N, R and W?"

controller = DocumentController()
controller.process_text_and_store(pdf_parser.get_text_from_pdf())
context = controller.search_and_retrieve_result(query)

generate = ChatOpenAI()
print(generate.get_message(query, context))
