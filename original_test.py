import torch
import os
import time
from transformers import pipeline
print("start up")
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'



model_dir = "./llama/llama-2-70b-chat-hf"
t1 = time.perf_counter()


generator = pipeline(task="text-generation", model=model_dir, torch_dtype=torch.bfloat16, device_map="auto")

t2 = time.perf_counter()
print(f"Loading tokenizer and model : took {t2 - t1} seconds to execute")


input_text = "Hello my dog is cute and"
output = generator(input_text)
print(output)




t3 = time.perf_counter()
print(f"Generating anwser : took {t3 - t2} seconds to execute")





