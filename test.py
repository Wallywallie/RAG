

import torch
import transformers
import os
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from accelerate import load_checkpoint_and_dispatch,init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
cuda_list = '0, 1, 2, 3'.split(',')
memery = '42GiB'
max_memory = {int(cuda): memery for cuda in cuda_list}

model_dir = "./llama/llama-2-70b-chat-hf"
model_dir2 = "./llama/llama-2-7b-hf"

t1 = time.perf_counter()

no_split_module_classes = LlamaForCausalLM._no_split_modules
config = LlamaConfig.from_pretrained(model_dir)
with init_empty_weights():
    model = LlamaForCausalLM._from_config(config, torch_dtype = torch.float16)

no_split_module_classes = ["Linear", "LlamaAttention", "LlamaDecoderLayer", "LlamaRMSNorm","SiLU","LlamaMLP","LlamaRotaryEmbedding","LlamaAttention" ]

device_map = infer_auto_device_map(model, max_memory = max_memory, no_split_module_classes = no_split_module_classes)   



load_checkpoint_in_model(model, model_dir, device_map= device_map) 



model = dispatch_model(model, device_map=device_map)

tokenizer = LlamaTokenizer.from_pretrained(model_dir)
torch.set_grad_enabled(False)
model.eval()
#print(model)
#print(device_map)

for (name, module), (name, param) in zip(model.named_modules(), model.named_parameters()):
   print(f"{module} -> {param.device}")


"""
for (name, module), (name, param) in zip(model.named_modules(), model.named_parameters()):
    print(name + "|")
    print(str(type(module)) + "|")
    print(str(param.device))
    print("\n")
"""



t2 = time.perf_counter()
print(f"Loading tokenizer and model : took {t2 - t1} seconds to execute")





while True:
    print("\n=================PLEASE TYPE IN YOUR QUESTION==============\n")
    user_content = input("\nQuestion: ")
    user_content.strip()

    t4 = time.perf_counter()

    test = ['who you are']
    ids = tokenizer(test , max_length = 400, truncation = True, return_tensors = "pt")
    ids = ids.to(model.device) 
    outputs = model.generate(**ids, do_sample = False)
  

    t5 = time.perf_counter()
    print(f"Generating anwser : took {t5 - t4} seconds to execute")
    print(outputs)
    


