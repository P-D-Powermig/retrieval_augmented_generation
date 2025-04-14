import os
import sys

import constants
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
# from langchain_community.llms import openai
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.chatgpt import ChatGPTLoader

os.environ["OPENAI_API_KEY"] = constants.APIKEY

chat_model = ChatOpenAI(openai_api_key = constants.APIKEY)

# result = chat_model.in("Olá, tudo bem?")

# print(result)
query = sys.argv[1]
print(query)

# # loader = TextLoader('..\data\data.txt')
loader = ChatGPTLoader(log_file="..\data\data.txt")
loader.load()
# loader = DirectoryLoader('.', glob="*.txt")

embeddings = OpenAIEmbeddings()
index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])
# index = VectorstoreIndexCreator().from_loaders([loader])


response = index.query(query)
print(response)

