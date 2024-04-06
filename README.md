# PDF question and answering bot using LLMs

This is WIP.

This repo is to implement the core components of a RAG pipeline from scratch to understand the various aspects of a good RAG system.


## What does it have right now ?

Q&A for pdfs using various embeddings(local, API), text splitters, retrieval strategies, retrieval sources and LLMs like OpenAI, Claude and Perplexity.


## Tasks and General Notes

- Modularize Code ✅
- Introduce classes ✅
- Integrate other LLMs ✅ (OpenAI, Claude, Perplexity added. Gimme $5 I'll add Gemini as well)
- Improve Parsing.
  - Parsing instructions in LlamaParse? ✅ (Custom instructions in LlamaParse work exceedingly well!)
  - User is probably ok in waiting for a few extra seconds to ingest data ✅ (LlamaParse custom instruction takes care of this)
  - For research paper indexing, the references can be avoided while indexing ✅ (LlamaParse custom instruction takes care of this)
  - Add option to use pypdf instead of Llama Parse. Will reduce external dependency.
- Chunking (Or text splitting)
  - Different Characters. ✅ (Using a fullstop(period) is the only one that makes sense. Not good kinda lame)
  - MarkdownSplitter by Langchain ✅ (Seems to be working the best. Play around with chunk_size to get good results)
  - RecursiveSplitter by Langchain ✅ (Works pretty well. General advise seems to be to use this by default.)
  - [Other Langchain Splitters](https://api.python.langchain.com/en/latest/text_splitters_api_reference.html)
  - AI21 chunker - This is a semantic text splitter available through API. ✅
    - Notes : 
      - The original ordering of the document is lost(which is ok, I guess).
      - The chunk size is only useful when the chunk size created is smaller than the chunk_size entered as the parameter. In my experience, this is not very useful as I usually want more precision-hence smaller chunks, combined with sentence window retrieval.
      - Sentence window retrieval means something entirely different for this chunker. It still works decently since you can expect some related text around the selected text. But not always the case. 
    - Pros :
      - It does a good job of semantically chunking text.
    - Cons :
      - External Dependency. Always gotta be a bit careful with those.
  - A big brain move - Use LLM for summarizing and chunking the text. Probably turns out to be expensive but data quality would be MUCH better. 
    - Could be effective with 7B param LLMs deployed on prem(No API business).
    - Context window sizes will be a problem.
  - Some other heuristic splitters.
- Context Retrieval Strategy
  - Closely related to Chunking.
  - Challenge : Create a chunking + storage strategy that is agnostic of retrieval strategy and storage medium(vectorDB, localStore).✅
    - Challenge Accepted : Created POC of a linked list based object store. Implemented Auto Merge Retrieval and Sentence Window Retrieval. POC is [here](https://github.com/ujjwalm29/pdf-reader/blob/main/ingestion/chunking/ChunkingController.py).✅   
  - Basic Context Retrieval ✅
  - Sentence Window Retrieval ✅
  - Auto Merging Retrieval (Tree Based) (Implemented using chunking+storage strategy)✅
- Core Retrieval
  - Embeddings options 
  - Store embeddings efficiently.
  - Integrate Vector DBs.
  - Option for attaching Postgres/Elasticsearch instance for search.
  - Advanced retrieval techniques (2 stage, IVFOPQ, int8 vectors)
  - Adaptive retrieval using Matryoshka Representations. [Reference](https://ujjwalm29.medium.com/matryoshka-representation-learning-a-guide-to-faster-semantic-search-1c9025543530) 
- Features 
  - Multiple PDFs (should be easy, just repeat everything for each file)
- Deployment and Productization 
  - Logging using loggers and NOT print statements lol.
  - Create Streamlit UI
  - Create Dockerfile
  - Deploy somewhere
  - Tests :(
- PI removal 
- Evaluation using Tru Lens
- Multi-Query(Query Expansion)
  - Old school, semantic Knowledge graphs created using SpaCy or leveraging inverted+forward index.
    - Can create alternate words, but how to create sentences?
  - Using LLMs : Feed the query to an LLM, ask it to expand on it and give 3 alternatives. Search and aggregate results.



## Articles/Libraries to write?

- Build a proxy for multiple LLM API providers?
- How to use llama parse effectively?
- A guide to chunking strategies
- 