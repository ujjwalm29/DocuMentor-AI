# PDF question and answering bot using LLMs

This is WIP.

This repo is to implement the core components of a RAG pipeline from scratch to understand the various aspects of a good RAG system.


## Features

Q&A for pdfs.
- How does the PDF get parsed? This step is called Parsing.
  - Using LlamaParse by LlamaIndex.
  - Pypdf3 can also be used.
- We need to store the file contents somehow. Can't store the full file as it's not efficient for search and retrieval. How is the text divided? This step is called chunking/splitting.
  - LangChain framework has several text splitters like MarkdownSplitter, Recursive Splitter etc.
  - Can develop your own splitter by using regex.
  - Can use AI21 splitter via API.
- How do we store and get documents? This is the core retrieval part.
  - Use a simple pandas Dataframe and store it in a pickle file for persistence storage.
  - Use a vector Database - Weaviate added, Vectara soon to be added.
  - Use a traditional search engine like Elasticsearch, Solr, Opensearch.
  - The lines between Vector DB and the Elastic, Solr etc are blurry. Both do pretty much the same thing.
- When we get some top N search results for a query, how do we use the results? This is the context retrieval strategy.
  - Simple Retrieval Strategy : Use the top N results and pass it to the the next stage.
  - Sentence window retrieval : If a certain piece of text matches our query, it is possible that the sentences surrounding that text will also be relevant as context. So, fetch results. Then fetch the text around the retrieved results.
  - Auto Merging Retrieval : In this structure, the chunks are in a tree structure. There are parent chunks, each having N(usually 4) child chunks. If the retrieved results contain >= N/2 children of a particular parent, it is likely that the parent block in itself is relevant to the query.
- Challenge : How do you build a framework where it doesn't matter whether you're using a Dataframe, Vector DB, Solr etc and it doesn't matter what context retrieval strategy is used?
  - LlamaIndex and Langchain do this sort of out-of-the-box.
  - I implemented my own linked list based strategy which can be used to solve this.


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
  - Challenge : Create a chunking + storage strategy that is agnostic of retrieval strategy and storage medium(vectorDB, localStore).✅
    - Challenge Accepted : Created POC of a linked list based Parent and child object store. Implemented Auto Merge Retrieval and Sentence Window Retrieval. POC is [here](https://github.com/ujjwalm29/pdf-reader/blob/main/ingestion/chunking/ChunkingController.py).✅   
  - Basic Context Retrieval ✅
  - Sentence Window Retrieval ✅
  - Auto Merging Retrieval (Tree Based) (Implemented using chunking+storage strategy)✅
- Core Retrieval
  - Embeddings
    - Create Embeddings locally ✅ - HuggingFace sentence-transformers library
    - Create embeddings through API ✅ - OpenAI embeddings supported.
    - Store embeddings efficiently.
    - Advanced retrieval techniques (IVFOPQ, int8 vectors, binary quantization) ❌- Not really needed here. Examples available in [my retrieval guide.](https://github.com/ujjwalm29/movie-search/tree/master/level_6_faiss_IVFOPQ_HNSW)
    - Adaptive retrieval using Matryoshka Representations.❌- Not really needed here. Example given in my [medium](https://ujjwalm29.medium.com/matryoshka-representation-learning-a-guide-to-faster-semantic-search-1c9025543530) article.
  - Integrate Vector DBs. ✅ - Added Weaviate. About to get Vectara credits, will add soon
    - Issue ❗- Need to build an abstraction such that getting results from database or pkl file should be easily swappable. ✅ - DONE
    - Issue ❗- Need to convert chunking strategy into an abstraction and implementable for dataframe, vector DB etc ✅ - DONE
  - Use Hybrid search ✅ - Through weaviate. Abstraction pending
  - Option for attaching Postgres/Elasticsearch(or Solr, Opensearch etc) instance for search.❌
    - Is it possible to do search using SQL? [Levels.fyi did it](https://www.levels.fyi/blog/scalable-search-with-postgres.html)
    - MOST people usually move things to Elasticsearch for keyword search.
    - Hence, if you're using SQL, move data to a search engine(like Solr etc) which can do good hybrid search.
    - I am aware of Elasticsearch basics, will add Elasticsearch integration
  - Design Flaw Maybe: In the storage class, a lot of if conditions getting created on the basis of index_name(parent, child). Is there a way to unify in 1 single index? ✅
    - The child chunk and parent chunk classes have been unified in the branch `schema_change`. But there are a few issues :
      - If you put everything into 1 index, then parent chunks become part of the search. This is unnecessary and will affect performance.
      - An option would be to add a Filter on the chunks with 0 children and then do a search. That would add unnecessary overhead as well.
- Features 
  - Multiple PDFs (should be easy, just repeat everything for each file) ✅ - Call the API/function
- Deployment and Productization 
  - Logging using loggers and NOT print statements. ✅ - `INFO`, `DEBUG` and `WARN` logging added 
  - Create APIs for operations
    - Add document - ✅
    - Search - ✅
    - Delete Indexes - ✅
    - Add auth
      - Added hacky API key based auth - ✅
    - Remove document?
      - Each doc indexed needs to have an ID. Each chunk needs a doc ID. Enter doc ID to delete - ✅
  - Create Streamlit UI
  - Storage should be a singleton dependency - ✅
  - Deployment stuff
    - Docker  ✅
    - docker compose  ✅
  - What if multiple users use it?
    - Added user_id to the schema - ✅
  - Tests :(
  - Error Handling
- PI removal 
- Evaluation using Tru Lens
- Multi-Query(Query Expansion)
  - Old school, semantic Knowledge graphs created using SpaCy or leveraging inverted+forward index.
    - Can create alternate words, but how to create sentences?
  - Using LLMs : Feed the query to an LLM, ask it to expand on it and give 3 alternatives. Search and aggregate results.
- Summarization pipeline
  - Sometimes, people just want summaries of pdfs.
  - Use OpenAI GPT 4 Turbo with batch functionality
    - Store the file in OpenAI storage
    - Create a batch with summary prompt - Store the batch ID somewhere
    - Create a cron job like process(is that a webhook?) to check whether job is complete.
    - Store summary in separate index with doc_id as key.
    - Find a way to decide whether user is asking for summary or a specific question. (Use an LLM?)
- How does one build a chat interface over RAG?
  - Let's say, I ask a question and get an answer. What's next?
  - Is the next prompt considered a new question? Do we do retrieval -> generation?
  - What if it's a follow up to the answer? Like "explain the previous answer like I'm 5"?



## Articles/Libraries to write?

- Build a proxy for multiple LLM API providers?
- How to use llama parse effectively?
- A guide to chunking strategies
- Write about chunking POC.
- Write about each section
- 