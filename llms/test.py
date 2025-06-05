import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

DEVICE = "cuda:6"

llm_path = "Baichuan2-13B-Base"
# tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True).to(DEVICE)
generator = pipeline("text-generation", model=llm_path, tokenizer=llm_path, trust_remote_code=True, device=DEVICE)

text = "今天天气不错"
result = generator(text, max_length=50, do_sample=True, temperature=0.8)
print(result)

