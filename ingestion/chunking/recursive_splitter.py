from langchain.text_splitter import RecursiveCharacterTextSplitter
from ingestion.chunking.text_splitter import TextSplitter


class RecursiveTextSplitter(TextSplitter):

    def __init__(self, chunk_size: int = 500, overlap_size: int = 0):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def split_text(self, text: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size
        )

        return text_splitter.split_text(text)



