from langchain.text_splitter import MarkdownTextSplitter
from ingestion.splitters.text_splitter import TextSplitter


class MarkdownTextParser(TextSplitter):

    def __init__(self, chunk_size: int = 500, overlap_size: int = 0):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def split_text(self, text: str):
        semantic_text_splitter = MarkdownTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap_size)
        chunks = semantic_text_splitter.split_text(text)

        return chunks
