

import torch
import transformers
import os
import time
from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import load_checkpoint_and_dispatch,init_empty_weights

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

model_dir = "./llama/llama-2-70b-chat-hf"
model_dir2 = "./llama/llama-2-7b-hf"

t1 = time.perf_counter()
model = LlamaForCausalLM.from_pretrained(model_dir)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

t2 = time.perf_counter()
print(f"Loading tokenizer and model : took {t2 - t1} seconds to execute")

pipeline = transformers.pipeline(
"text-generation",
model=model,
tokenizer=tokenizer,
torch_dtype=torch.float16,
device_map="auto",
)

t3 = time.perf_counter()
print(f"Creating pipeline : took {t3 - t2} seconds to execute")

while True:
    print("\n=================PLEASE TYPE IN YOUR QUESTION==============\n")
    user_content = input("\nQuestion: ")
    user_content.strip()

    t4 = time.perf_counter()

    sequences = pipeline(
    user_content,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    )

    t5 = time.perf_counter()
    print(f"Generating anwser : took {t5 - t4} seconds to execute")
    
    for seq in sequences:
        print(f"{seq['generated_text']}")

