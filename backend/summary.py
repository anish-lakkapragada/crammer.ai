import os
import langchain
from langchain_ai21 import ChatAI21
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ai21 import AI21Embeddings, AI21ContextualAnswers
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool

os.environ["AI21_API_KEY"] = "tQ5ybFtCPDfMCgXZiCdVLwejMyKYro8p"

loader = DirectoryLoader(
    path = "./videos",
    glob="**/*.txt",
    show_progress = True
)

documents = loader.load()

concat_docs = ""
for doc in documents:
    concat_docs += doc.dict()['page_content']



template = """
Given this context: {context}

Please do: {question}
"""

prompt = PromptTemplate.from_template(template)

model = ChatAI21(model="jamba-instruct", max_tokens=500, streaming=True)

chain = prompt | model

response = chain.invoke({"context":concat_docs, "question": "Please give a brief, one sentence summary."}).dict()["content"]

print(response)