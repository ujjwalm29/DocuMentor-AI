import os

from ingestion.pdf import parse_pdf
from chunking.naive_text_splitter import split_text
from embeddings.local import LocalEmbeddings
from retrieval.local_dataframe import sentence_window_retrieval
from generation.openai_chat import generate_response

from dotenv import load_dotenv
import pandas as pd


load_dotenv()
embeddings = LocalEmbeddings()

MD_FILE_NAME = "data/markdowns/amazon-dynamo-sosp2007.md"


# check if MD_FILE_NAME exists
if not os.path.exists(MD_FILE_NAME):
    documents = parse_pdf('data/pdf/amazon-dynamo-sosp2007.pdf')

# documents is a md file. write documents to disk
    with open(f"./{MD_FILE_NAME}", "w") as file:
        file.write(documents[0].text)

# read pickle file called df_with_embeddings.pkl
if os.path.exists("data/pkl/df_with_embeddings.pkl"):
    df = pd.read_pickle("data/pkl/df_with_embeddings.pkl")
else:
    # read the md file
    with open(f"./{MD_FILE_NAME}", "r") as file:
        document = file.read()

    # Split document into sentences
    sentences = split_text(document)

    # Initialize DataFrame
    df = pd.DataFrame(sentences, columns=['sentence'])

    # Apply function across DataFrame
    df = embeddings.get_embeddings_for_dataframe(df)

    # Save to pickle file
    df.to_pickle("df_with_embeddings.pkl")

query = "Does Dynamo handle reads better or writes?"

query_embedding = embeddings.get_embedding([query])

final_results = sentence_window_retrieval(df, query_embedding)

print(generate_response(query, final_results))
