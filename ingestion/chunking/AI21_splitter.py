from langchain_ai21 import AI21SemanticTextSplitter
from ingestion.chunking.text_splitter import TextSplitter


class AI21TextParser(TextSplitter):

    def split_text(self, text: str):
        semantic_text_splitter = AI21SemanticTextSplitter(chunk_size=400)
        chunks = semantic_text_splitter.split_text(text)

        return chunks
