# PDF question and answering bot using LLMs

This is WIP.

All of this can be done using LlamaIndex in far fewer lines of code.
This repo is to implement the core components of a RAG pipeline.


## What does it have right now ?
- Q&A for 1 pdf using local embeddings, brute force retrieval and openAI LLMs.
- Raw implementation of chunking + sentence window retrieval to get context for user query.
- 




## What else can I do with this?

- ~~Modularize Code~~
- ~~Introduce classes~~
- ~~Integrate other LLMs~~ (OpenAI, Claude, Perplexity added. Gimme $5 I'll add Gemini as well)
- Improve Parsing.
  - User is probably ok in waiting for a few extra seconds to ingest data
  - Parsing instructions in LlamaParse?
  - For research paper indexing, the references can be avoided while indexing
- Chunking 
  - Different Characters
  - Check LangChain splitters.
  - Some other heuristic splitters?
- Multiple PDFs
- Store embeddings efficiently.
- Embeddings options
- Create Streamlit UI
- Create Dockerfile
- Deploy somewhere
- Evaluation using Tru Lens
- Tests :(
- Option for attaching Postgres/Elasticsearch instance for search.
- Integrate Vector DBs.
- Advanced retrieval techniques (2 stage, IVFOPQ, int8 vectors)