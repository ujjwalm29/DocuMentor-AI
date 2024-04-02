from langchain.text_splitter import CharacterTextSplitter


def split_text(text: str, chunk_size: int = 200, overlap_size: int = 10):
    text_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_text(text)



