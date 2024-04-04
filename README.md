# PDF question and answering bot using LLMs

This is WIP.

All of this can be done using LlamaIndex in far fewer lines of code.
This repo is to implement the core components of a RAG pipeline.


## What does it have right now ?
- Q&A for 1 pdf using local embeddings, brute force retrieval and openAI LLMs.
- Raw implementation of chunking + sentence window retrieval to get context for user query.
- 




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
  - Challenge : Create a chunking + storage strategy that is agnostic of retrieval strategy 
  - Basic Context Retrieval ✅
  - Sentence Window Retrieval ✅
  - Auto Merging Retrieval (Tree Based)
- Core Retrieval
  - Embeddings options 
  - Store embeddings efficiently.
  - Integrate Vector DBs.
  - Option for attaching Postgres/Elasticsearch instance for search.
  - Advanced retrieval techniques (2 stage, IVFOPQ, int8 vectors)
- Features 
  - Multiple PDFs (should be easy, just repeat everything for each file)
- Deployment and Productization 
  - Create Streamlit UI
  - Create Dockerfile
  - Deploy somewhere
  - Tests :(
- PI removal 
- Evaluation using Tru Lens
- Multi-Query



## Articles/Libraries to write?

- Build a proxy for multiple LLM API providers?
- How to use llama parse effectively?
- A guide to chunking strategies
- 