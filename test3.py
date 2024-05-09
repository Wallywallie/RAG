import torch
import transformers
import os
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from accelerate import load_checkpoint_and_dispatch,init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model

model_dir = "./llama/llama-2-70b-chat-hf"
config = LlamaConfig.from_pretrained(model_dir)
no_split_module_classes = LlamaForCausalLM._no_split_modules

with init_empty_weights():
    model = LlamaForCausalLM._from_config(config, torch_dtype = torch.float16)
model.tie_weights()

model = load_checkpoint_and_dispatch(
    model, 
    model_dir, 
    device_map="auto", 
    no_split_module_classes=no_split_module_classes)

device = torch.device("cuda")
model.cuda()

print(model.hf_device_map)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
inputs = tokenizer("hello", return_tensors = "pt")
inputs = inputs.to(device)

output = model.generate(inputs["input_ids"])
print(output)