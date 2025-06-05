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
   
##############################################################################
    homograph_path = "homoGraphs/COLD_homoGraph"
    data_path = "datas/COLD/COLD_key.json"
    output_path = "datas/processed_data/COLD/"

    for llm_name in llm_names:
        llm_model = LLM_model(llm_configs[llm_name], device=DEVICE)
        bert_model = Bert_mask_model(
            bert_configs[bert_name]["path"], device=DEVICE)
        detector = DetectorModel(homograph_path)
        denoiseModel = DenoiseModel(llm_model, bert_model, detector)
        a = 0
        start_time = time.time()
        with open(data_path, "r", encoding="utf-8") as f:
            datas = json.loads("".join(f.readlines()))
            N = len(datas)
            for mode in denoiseModel.modes:
                # print(mode)
                # if not "llm-m" in mode:
                #     continue
                dataset_name = "dn"
                if "llm" in mode:
                    dataset_name += "_%s" % llm_name
                dataset_name += "_%s" % mode
                output_file = output_path + dataset_name + ".json"
                new_datas = []
                a = 0
                for i, data in enumerate(datas):
                    print(" %d / %d" % (i, N), end="\r")
                    text, label = data["text"].strip(" \n\'\""), data["label"]
                    new_text = denoiseModel.denoise(text, mode)
                    # print(new_text)
                    # new_datas.append({"text": new_text, "label": label})
                # with open(output_file, "x", encoding="utf-8") as _of:
                #     _of.write(json.dumps(new_datas, ensure_ascii=False))
                # print("\nSaved denoised file %s" % output_file)

        del llm_model, bert_model, detector, denoiseModel

    end_time = time.time()

    print("\nTotal time llm-m COLD: %.2f s" % (end_time - start_time))
