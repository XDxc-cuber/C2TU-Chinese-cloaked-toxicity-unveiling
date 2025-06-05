from models.llm import LLM_model
from models.bert import Bert_mask_model
from models.lexicon_detector import DetectorModel
from models.denoiseModel import DenoiseModel

import time

import json


if __name__ == "__main__":
    llm_configs, bert_configs = {}, {}
    with open("llms/llm_configs.json", "r") as f:
        llm_configs = json.loads("".join(f.readlines()))
    with open("berts/bert_configs.json", "r") as f:
        bert_configs = json.loads("".join(f.readlines()))
    
    DEVICE = "cuda:5"
    llm_names = ["baichuan7"]
    bert_name = "bert-base-chinese"
    
    llm_name = llm_names[0]
   
    homograph_path = "homoGraphs/ToxiCN_homoGraph"

    llm_model = LLM_model(llm_configs[llm_name], device=DEVICE)
    bert_model = Bert_mask_model(
        bert_configs[bert_name]["path"], device=DEVICE)
    detector = DetectorModel(homograph_path)
    denoiseModel = DenoiseModel(llm_model, bert_model, detector)
    
    mode = "llm-m"
    text = "你这个沙币，继续舔你的嘿贵去吧！"
    new_text = denoiseModel.denoise(text, mode)
    
    print("Input: %s\nOutput: %s" % (text, new_text))
