#main

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma

from langchain import LLMChain
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

#import qianfan
import chromadb
import os
import re
#import pdfplumber
import time
import prompt_template

from langchain.document_loaders import PDFPlumberLoader
import query
import json

chunk_size = 768
chunk_overlap = 0
question = "两栋夹角为60度的居住建筑的间距应该怎样控制？"#"两栋平行的居住建筑的间距应该怎样控制？"
db_path = "./database"
folder_path = "..\\data"
file_path = "./database/test.txt"
load_new = False
embeddingmodel_path = "./model/m3e-base"
model = "ERNIE-Lite-8K-0308"
test_cnt = 1


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


if __name__ == '__main__':

    embedding_func = embedding_function() 
    db = load_db(db_path, embedding_func)

    if load_new:
        documents = load_txt(file_path)
        split_documents = split_data(documents)
        db = embedding(split_documents, embedding_func, db)

    
    llm = QianfanChatEndpoint(model= model, qianfan_ak ="sXu3ML0Q1VitaW9X98Zo2yJQ", qianfan_sk ="pYEw0w3kBLbBT8CDLqZR0NpoNbxmXzi4") #"Meta-Llama-3-70B" ERNIE-Lite-8K-0922 ERNIE-3.5-8k-Preview
    retriever = db.as_retriever()
    prompt = prompt_template.prompt_test
    output_parser = StrOutputParser()
    setup_and_retriever = RunnableParallel({"context" : retriever,"instruction": RunnablePassthrough(),"input": RunnablePassthrough()})



    

    with open('question.json', 'r', encoding='utf-8') as file:
        test = json.load(file)

    


    fname = model + ".json"
    with open(fname, 'a', encoding = 'utf-8') as result_f:
        result_f.write("[\n")
    for q in test:
        
        prompt1 = prompt.partial(instruction = q["instruction"])
        retrieve_data = db.similarity_search(q["question"])
        data = {"question":q["question"],"retrieve":str(retrieve_data),"retrieve_len": len(str(retrieve_data)),"result": [], "time":[],"len":[],"score":[],"average_time&len&score":[]}
        for i in range(test_cnt):
            
            chain = setup_and_retriever | prompt1 | llm | output_parser
            t1 = time.perf_counter()
            output = chain.invoke(q["question"])
            t2 = time.perf_counter()
            generate_time = t2-t1
            data["result"].append(output)
            data["time"].append(generate_time)
            data["len"].append(len(output))
    
        
        with open(fname, 'a', encoding = 'utf-8') as result_f:
            json.dump(data, result_f, indent=4,ensure_ascii=False)
            result_f.write(',\n')
        break
            
        
  

with open(fname, 'a', encoding = 'utf-8') as result_f:
    result_f.write("]\n")
print("DONE")
            
    





    

   

