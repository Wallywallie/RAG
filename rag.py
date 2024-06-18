from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma


from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


import torch
import os
import time
from transformers import pipeline

import chromadb
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.retrieval_qa.base import VectorDBQA
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
import prompt_template
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
import query

chunk_size = 256
chunk_overlap = 0
question = "两栋平行的居住建筑的间距应该怎样控制？"
db_path = "./UrbanPlanningDemo/database"
folder_path = "..\\data"
file_path = "../data2/test.txt"
load_new = False
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
model_dir = "./llama/llama-2-70b-chat-hf"
#model_dir = "./llama/llama-2-7b-hf"
embeddingmodel_path = "./model/m3e-base"



#本地数据加载
def load_txt(file_path : str):
    loader = TextLoader(file_path, "utf-8")
    document = loader.load()
    #print(document)
    return document

#文档分割
def split_data(document):
    text_splitter = RecursiveCharacterTextSplitter(separators = ['\n\n','\n', '。'],chunk_size = chunk_size, chunk_overlap = chunk_overlap) 
    #text_splitter = CharacterTextSplitter(separator = '\n',chunk_size = chunk_size, chunk_overlap = chunk_overlap) 
    #CharacterTextSplitter的默认分割符号是\n\n换行符
    #chunk_size 每个chunk的最大长度
    #chunk_overlap 每个chunk之间的重叠程度
    documents = text_splitter.split_documents(document)
    print(len(documents))
    return documents


#加载数据库
def load_db(db_path, embedding):
    if os.path.isfile(db_path + "/chroma.sqlite3") is not True:
        db = chromadb.PersistentClient(db_path)
        print("-------database has been created-------")
    else:
        db = Chroma(persist_directory=db_path, embedding_function = embedding)
        print(db)
        print("-------loading database-------")
    return db    


#嵌入方式
def embedding_function():
    # embedding model: m3e-base
    model_name = embeddingmodel_path
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    query_instruction="为文本生成向量表示用于文本检索"
                )
    return embedding

#向量化
def embedding(documents, embedding, db):

    # load data to Chroma db
    print("-------embedding data... -------")
    db = Chroma.from_documents(documents, embedding, persist_directory=db_path)
    print("-------saving database -------")
    return db


embedding_func = embedding_function() 
db = load_db(db_path, embedding_func)

if load_new:

    documents = load_txt(file_path)
    split_documents = split_data(documents)
    db = embedding(split_documents, embedding_func, db)    


generator = pipeline(task="text-generation", model=model_dir,torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens = 1024)

pipeline = HuggingFacePipeline(pipeline = generator)
retriever = db.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = prompt_template.prompt

output_parser = StrOutputParser()
setup_and_retriever = RunnableParallel({"context" : retriever, "input": RunnablePassthrough()})


chain = setup_and_retriever | prompt | pipeline | output_parser

if __name__ == "main":
    t1 = time.perf_counter()
    response = chain.invoke(query.l3)
    t2 = time.perf_counter()
    print(response)
    print(f"it takes {t2-t1} to generate")

