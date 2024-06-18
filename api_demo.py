import requests
from langchain.llms.base import LLM
from typing import Optional
from typing import List
import prompt_template
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
import os
import chromadb
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
import time
db_path = "./UrbanPlanningDemo/database"
#question = "两栋平行的居住建筑的间距应该怎样控制？"
question = "两栋夹角为60度的居住建筑的间距应该怎样控制？"
embeddingmodel_path = "./model/m3e-base"
class MyLocalModel(LLM):
    api_url: str
    def __init__(self, api_url: str):
        super().__init__(api_url=api_url)
        #self.api_url = api_url

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # 将prompt和stop参数转化为适合API的格式
        #data = {"model": "llama3:70b","prompt": prompt,"stream": False,"format": "json"}
        data = {
            "model": "llama3:70b",#qwen:110b
            "prompt": prompt,
            "stream": False
            }
        if stop:
            data["stop"] = stop
        try:
            # 发送POST请求到本地API
            response = requests.post(self.api_url, json=data)
            response.raise_for_status()  # 如果响应状态不是200，会抛出HTTPError
            response_json = response.json().get("response","")
            return response_json
        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"Something went wrong: {err}")
        if response.status_code == 200:
            try:
                result = response.json().get("response","")
                return result
            except ValueError as ve:
                print(f"Error decoding JSON: {ve}")
                print(f"Raw response response: {response.response}")
        else:
            raise Exception(f"Model service returned an error: {response.response}")
    def _llm_type(self):
        return "my_local_model"

    @property
    def _type(self):
        return "my_local_model"

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

embedding_func = embedding_function() 
db = load_db(db_path, embedding_func)
retriever = db.as_retriever()
# 假设你的API地址是 http://internal.shanhaiengine.com:11434/api/generate
api_url = "http://internal.shanhaiengine.com:11434/api/generate"
llm = MyLocalModel(api_url)


# 使用LangChain创建一个链
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 正确设置模板，假设你想在问候语中插入名字
#template = "are you human? {name}!"
#prompt = PromptTemplate(input_variables=["name"], template=template)
prompt = prompt_template.prompt
output_parser = StrOutputParser()
setup_and_retriever = RunnableParallel({"context" : retriever, "input": RunnablePassthrough()})


chain = setup_and_retriever | prompt | llm | output_parser

# 创建链，使用上面定义的prompt
#chain = LLMChain(llm=llm, prompt=prompt)

# 运行链，传递名字作为输入
# 注意，这里应该传递一个字典，键是模板中定义的变量名
#name_input = {"name": "Alice"}  # 以字典形式提供输入
t1 = time.perf_counter()
output = chain.invoke(question)
t2 = time.perf_counter()
print(f"it takes {t2-t1}\n")
print("======================================================result here :", output)
