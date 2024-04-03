from ingestion.pdf import PdfParser
from embeddings.local import LocalEmbeddings
from retrieval.local_dataframe import LocalDataframe
from retrieval.sentence_window import SentenceWindowRetrieval
from generation.openai_chat import ChatOpenAI
from generation.claude_chat import ChatClaude
from generation.pplx_chat import ChatPplx

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

file_path = 'data/pdf/amazon-dynamo-sosp2007.pdf'
pkl_file_name = ''.join(file_path.split('/')[-1].split('.')[:-1]) + '.pkl'
pkl_file_path = os.path.join('data', 'pkl', pkl_file_name)

embeddings = LocalEmbeddings(file_path=file_path, embeddings_model="mixedbread-ai/mxbai-embed-large-v1")
pdf_parser = PdfParser(file_path=file_path, parsing_instructs="""
You are parsing a research paper. DO NOT parse or include the references section in the output. 
Convert tables into a list of facts. 
""")

if not os.path.exists(pkl_file_path):
    df = pdf_parser.parse_pdf()
    df = embeddings.get_embeddings_for_dataframe(df)
else:
    df = pd.read_pickle(pkl_file_path)

query = "What are some problems with dynamo?"
query_embedding = embeddings.get_embedding([query])

retrieval = LocalDataframe(SentenceWindowRetrieval())

final_results = retrieval.get_results(df, query_embedding)

for result in final_results:
    print(result)

generation = ChatClaude()

print(generation.get_message(query, final_results))
