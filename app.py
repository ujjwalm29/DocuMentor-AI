from ingestion.pdf import PdfParser
from embeddings.local import LocalEmbeddings
from retrieval.local_dataframe import sentence_window_retrieval
from generation.openai_chat import generate_response

import os
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
file_path = 'data/pdf/amazon-dynamo-sosp2007.pdf'
pkl_file_name = ''.join(file_path.split('/')[-1].split('.')[:-1]) + '.pkl'
pkl_file_path = os.path.join('data', 'pkl', pkl_file_name)

embeddings = LocalEmbeddings(file_path=file_path, embeddings_model="mixedbread-ai/mxbai-embed-large-v1")
pdf_parser = PdfParser(file_path=file_path)

if not os.path.exists(pkl_file_path):
    df = pdf_parser.parse_pdf()
    df = embeddings.get_embeddings_for_dataframe(df)
else:
    df = pd.read_pickle(pkl_file_path)

query = "What are some problems with dynamo?"
query_embedding = embeddings.get_embedding([query])

final_results = sentence_window_retrieval(df, query_embedding)

for result in final_results:
    print(result)

print(generate_response(query, final_results))
