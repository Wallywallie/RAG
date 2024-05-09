from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from torch.cuda.amp import autocast
import torch

model_dir = "./llama/llama-2-70b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
 
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map='auto', torch_dtype=torch.float16)
 
 
no_split_module_classes = ["Linear", "LlamaAttention", "LlamaDecoderLayer", "LlamaRMSNorm","SiLU","LlamaMLP","LlamaRotaryEmbedding","LlamaAttention" ]
no_split_module_classes = ["LlamaDecoderLayer"]
 
max_memory = get_balanced_memory(
    model,
    max_memory=None,
    no_split_module_classes=no_split_module_classes,
    dtype='float16',
    low_zero=False,
)
  
device_map = infer_auto_device_map(
    model,
    max_memory=max_memory,
    no_split_module_classes=no_split_module_classes,
    dtype='float16'
)
 
model = dispatch_model(model, device_map=device_map)
 
generation_kwargs = {
    "min_length": -1,
    "top_k": 0,
    "top_p": 0.85,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "min_new_tokens": 10,
    "max_new_tokens": 50,
    "eos_token_id": tokenizer.eos_token_id,
}


 
with autocast():
    print(tokenizer.decode(model.generate(tokenizer.encode("Hello World!", return_tensors="pt").to(model.device), **generation_kwargs)[0]))