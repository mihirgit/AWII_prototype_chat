
"""
Author: Mihir Goyenka

This code is intended for a prototype chatbot for AWII

"""

import os, pickle
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
import faiss
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

os.environ["OPENAI_API_KEY"] = "API-KEY-HERE"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "API-KEY-HERE"


# URL for scraping info about AWII

# Additional task: scrape websites and gather links to generate more context data for chatbot

# using only 1 news article initially
urls = [
    'https://www.statepress.com/article/2023/04/arizona-desert-water-innovation-initiative'
]

#load the data from website
loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()

# Text splitter
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200) #play around with chunk sizes and overlaps
docs = text_splitter.split_documents(data)


# Embeddings
# Additional tasks: local embeddings (InstructorEmbedding) can be used, Open-Source (HuggingFace) can also be used
# Need to compare performance and cost metrics for best solution of embeddings

embeddings = OpenAIEmbeddings()

# VectorStore
vectorStore_openAI = FAISS.from_documents(docs, embeddings)

#dump to disk instead of recompute
with open('faiss_store_openai.pkl', 'wb') as f:
    pickle.dump(vectorStore_openAI, f)

with open('faiss_store_openai.pkl', 'rb') as f:
    VectorStore = pickle.load(f)

# define LLM: can use open-source llms as well (HuggingFace GPT2)
llm = OpenAI(temperature=0,model_name='')

# prompt and response
# Additional tasks: develop GUI/Web-based UI to accept user prompt
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

chain({"question: What is Arizona Water Innovation Initiative?"}, return_only_outputs=True)

