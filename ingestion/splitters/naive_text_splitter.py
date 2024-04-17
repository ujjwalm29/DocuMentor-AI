from langchain_text_splitters import CharacterTextSplitter
from ingestion.splitters.text_splitter import TextSplitter


class NaiveTextSplitter(TextSplitter):

    def __init__(self, chunk_size: int = 500, overlap_size: int = 0):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def split_text(self, text: str):
        text_splitter = CharacterTextSplitter(
            separator=".",
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap_size,
            length_function=len,
            is_separator_regex=False,
        )

        return text_splitter.split_text(text)



