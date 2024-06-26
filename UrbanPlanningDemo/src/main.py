#main

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

from langchain import LLMChain
from langchain_community.chat_models import QianfanChatEndpoint

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

import qianfan
import chromadb
import os
import re
import pdfplumber

from langchain.document_loaders import PDFPlumberLoader


chunk_size = 256
chunk_overlap = 0
question = "两栋平行的居住建筑的间距应该怎样布置？"
db_path = "../database"
folder_path = "..\\data"
file_path = "../data2/test.txt"
load_new = False

prompt_urbanist = """
    # Role: 城市规划师
    ## Profile
    - Author: author
    - Version: 0.1
    - Language: 中文
    - Description: 你是一个有多年工作经验的城市规划师。你非常熟知上海市的城市规划管理规定。
    ### Skill
    1. 你十分掌握土地开发和土地利用规划的方法，平衡城市发展的需求与土地资源的供给，确保土地利用的合理性和效益。
    2. 你十分擅长分析城市社会经济特征和发展趋势，了解居民需求和就业机会，为城市规划提供数据支持和决策参考。
    3. 你十分熟悉城市规划相关的法律法规和政策，包括国家、地方的规划法规以及土地使用政策，确保规划的合法性和可行性。
    4. 你非常熟悉城市发展的基本理论、原则和方法，包括城市发展模式、城市功能区划、可持续发展等。
    5. 你非常熟悉城市设计原则，包括公共空间设计、建筑外观、景观设计等，以创造宜居、宜人的城市环境。
    6. 你具备非常好的沟通与协调能力和团队协作能力，能够与政府部门、社区居民、开发商等多方合作，推动城市规划的实施。
    ## Rules
    1. 不要在任何情况下破坏角色。
    2. 不要说废话或编造事实
    3. 不要介绍自己
    ## Query:两栋平行的居住建筑的间距应该怎样布置？
    ## Initialization
    作为角色 <Role>, 严格遵守 <Rules>, 使用默认 <Language> 与用户对话，回答用户的<Query>。    
"""

#本地数据加载
def load_txt(file_path : str):
    loader = TextLoader(file_path, "utf-8")
    document = loader.load()
    #print(document)
    return document

def load_pdf(folder_path:str):
    pdf_loaders = []
    documents = []

    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:

            # 检查文件的后缀名是否为 .pdf
            if file.endswith(".pdf"):
                # 构造文件的绝对路径
                file_path = os.path.join(root, file)
                # 使用 PDFPlumberLoader 加载 PDF 文件
                try:
                    loader = PDFPlumberLoader(file_path)
                    pdf_loaders.append(loader)
                except Exception as e:
                    print(f"Error loading PDF loader '{file_path}': {e}")
                    
    for i in pdf_loaders:
        try:
            document = i.load()   
            documents.extend(document)   
        except Exception as e:
            print(f"Error loading PDF file '{file_path}': {e}")              

    return documents
    

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
    model_name = "moka-ai/m3e-base"
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


if __name__ == '__main__':

    embedding_func = embedding_function() 
    db = load_db(db_path, embedding_func)

    if load_new:

        documents = load_txt(file_path)
        split_documents = split_data(documents)
        db = embedding(split_documents, embedding_func, db)



    # LLM选型
    llm = QianfanChatEndpoint(model="ERNIE-Lite-8K-0922", qianfan_ak ="sXu3ML0Q1VitaW9X98Zo2yJQ", qianfan_sk ="pYEw0w3kBLbBT8CDLqZR0NpoNbxmXzi4")
    retriever = db.as_retriever()
    print("---------------------")
    print(retriever)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    print("---------------------")
    print(memory)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
    print("---------------------")
    print(qa)
    print("---------------------")
    print(dir(qa))
    qa({"question": question})
    print(qa)


    

   

