from lmcsc import LMCorrector
import torch
import json
import time

corrector = LMCorrector(
    model="/mnt/cfs_algo_bj/models/experiments/maxuchen/Baichuan",
    # Suggested to use the same model for model and prompted_model. In this way, we only need to load the model once.
    prompted_model="/mnt/cfs_algo_bj/models/experiments/maxuchen/Baichuan",
    # You can always use the default config file to disable the insert and delete operations.
    config_path="/mnt/cfs_algo_bj/workspace/maxuchen/C2TU-main/csc_models/simple-csc-main/configs/default_config.yaml",
    # the default torch_dtype is torch.float16, but it will lead unexpected errors when using Qwen2 or Qwen2.5 family models without flash-attn.
    torch_dtype=torch.bfloat16,
)


if __name__ == "__main__":
    data_path = "../../datas/COLD/COLD_shape_key.json"
    output_path = "../../datas/processed_data/COLD/shape_csc_simple_csc_7B.json"
    # data_path = "../../datas/ToxicloakCN/ToxiCN_shape_key.json"
    # output_path = "../../datas/processed_data/ToxicloakCN/shape_csc_simple_csc_7B.json"
    # data_path = "/mnt/cfs_algo_bj/workspace/maxuchen/C2TU-main/datas/Toxic_rest/Toxic_real_world.json"
    # output_path = "../../datas/processed_data/ToxicloakCN/real_world_csc_simple_csc_7B.json"
    
    # s = "今天天琪不错"
    # ss = corrector(s)
    # print(ss)
    # start_time = time.time()
    with open(data_path, "r", encoding='utf-8') as f:
        datas = json.loads(f.readlines()[0])
        N = len(datas)
        for i, data in enumerate(datas):
            print(" %d/%d" % (i, N), end="\r")
            text = data["text"]
            new_text = corrector(text)[-1][0]
            datas[i]["text"] = new_text
            # if len(text) != len(new_text):
            #     print(text)
            #     print(new_text)
        with open(output_path, "x", encoding="utf-8") as _of:
            _of.write(json.dumps(datas, ensure_ascii=False))
            
    # data_path = "../../datas/ToxicloakCN/Toxic_key.json"
    # output_path = "../../datas/processed_data/ToxicloakCN/csc_simple_csc_7B.json"

    # with open(data_path, "r", encoding='utf-8') as f:
    #     datas = json.loads(f.readlines()[0])
    #     N = len(datas)
    #     for i, data in enumerate(datas):
    #         print(" %d/%d" % (i, N), end="\r")
    #         text = data["text"]
    #         new_text = corrector(text)[-1][0]
    #         datas[i]["text"] = new_text
    #         if len(text) != len(new_text):
    #             print(text)
    #             print(new_text)
    #     with open(output_path, "x", encoding="utf-8") as _of:
    #         _of.write(json.dumps(datas, ensure_ascii=False))


    # end_time = time.time()
    # print("Time taken: ", end_time - start_time)
