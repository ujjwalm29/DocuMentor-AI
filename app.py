from ingestion.pdf import PdfParser
from ingestion.splitters.recursive_splitter import RecursiveTextSplitter
from embeddings.LocalHFEmbeddings import LocalEmbeddings

from ingestion.storage.weaviateDB import WeaviateDB

import os
import pandas as pd
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

# if not os.path.exists(pkl_file_path):
#     df = pdf_parser.parse_pdf()
#     df = embeddings.get_embeddings_for_dataframe(df)
# else:
#     df = pd.read_pickle(pkl_file_path)
#
# # query = "What are some things to keep in mind before using Dynamo?"
# # query = "What are some things problems with using Dynamo?"
# query = "What do the values N,R and W mean?"
# query_embedding = embeddings.get_embedding(query)
#
# # weaviate_storage.get_all_items()
# weaviate_storage = WeaviateDB()
# weaviate_storage.store(df)
# final_results = weaviate_storage.search_using_vector(query_embedding.tolist())
#
# for result in final_results:
#     print(result)
#
# final_results = weaviate_storage.search_hybrid(query, query_embedding.tolist(), "text")
#
# print()
# for result in final_results:
#     print(result)
#
# weaviate_storage.delete_collection()
# weaviate_storage.close_connection()

# retrieval = LocalDataframe(BaseRetrieval())
# query_embedding = embeddings.get_embedding([query])
# final_results = retrieval.get_results(df, query_embedding)
# print()
# for result in final_results:
#     print(result)

# generation = ChatOpenAI()
#
# print()
# print(generation.get_message(query, final_results))


# chunking_controller = ChunkingController1()
# chunking_controller.split_text(pdf_parser.get_text_from_pdf())
