import os
import time
import json
import math
import torch
import torch.multiprocessing as mp
import numpy as np
import random

from models.llm import LLM_model
from models.bert import Bert_mask_model
from models.lexicon_detector import DetectorModel
from models.denoiseModel import DenoiseModel


def remove_ratio(lst, ratio, random_state=None):
    if ratio <= 0:
        return lst.copy()
    if ratio >= 1:
        return []

    n = len(lst)
    k = int(ratio * n)
    if k == 0:
        return lst.copy()

    if random_state is not None:
        random.seed(random_state)

    del_indices = set(random.sample(range(n), k))
    return [elem for i, elem in enumerate(lst) if i not in del_indices]

def remove_edges_by_proportion(A, proportion, random_state=None):
    if proportion <= 0:
        return A.copy()
    if proportion >= 1:
        return np.zeros_like(A)

    n = A.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    mask = A[triu_i, triu_j] != 0
    edge_indices = np.where(mask)[0]
    num_edges = len(edge_indices)

    if num_edges == 0:
        return A.copy()

    rng = np.random.default_rng(random_state)
    remove_cnt = int(proportion * num_edges)
    remove_idx = rng.choice(edge_indices, size=remove_cnt, replace=False)

    i_remove = triu_i[remove_idx]
    j_remove = triu_j[remove_idx]

    A_new = A.copy()
    A_new[i_remove, j_remove] = 0
    A_new[j_remove, i_remove] = 0
    return A_new

def worker_process(gpu_id, data_chunk, llm_name, llm_config, bert_config, homograph_path, drop_edge=False, drop_lexicon=False, prob=1.0):
    device = f"cuda:{gpu_id}"
    print(f"Worker on {device} started, processing {len(data_chunk)} items.", flush=True)

    # 每个进程在指定的 GPU 上独立初始化模型
    llm_model = LLM_model(llm_config, device=device)
    bert_model = Bert_mask_model(bert_config["path"], device=device)
    detector = DetectorModel(homograph_path)


    # drop something
    # drop edges
    if drop_edge:
        detector.homoGraph.A = remove_edges_by_proportion(detector.homoGraph.A, prob, 42)

    # drop lexicons
    if drop_lexicon:
        detector.homoGraph.lexicons = remove_ratio(detector.homoGraph.lexicons, prob, 42)


    denoiseModel = DenoiseModel(llm_model, bert_model, detector)
    
    # 存储该块数据在所有 mode 下的处理结果
    chunk_results = {mode: [] for mode in denoiseModel.modes}
    
    for mode in denoiseModel.modes:
        # 如果需要过滤 mode，可以在这里取消注释
        if "llm-m" not in mode:
            continue
        
        N = len(data_chunk)
        for i, data in enumerate(data_chunk):
            # 为了避免控制台输出混乱，只让 GPU:0 打印进度
            if i % 5 == 0:
                print(f"[GPU:{gpu_id} | Mode: {mode}] Progress: {i} / {N}", flush=True)
                
            text, label = data["text"].strip(" \n\'\""), data["label"]
            new_text = denoiseModel.denoise(text, mode)
            chunk_results[mode].append({"text": new_text, "label": label})
            
    # 清理显存
    del llm_model, bert_model, detector, denoiseModel
    torch.cuda.empty_cache()

    return chunk_results

# -------------------------------------------------------------------
# 2. 主函数
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 【关键】在 PyTorch 中使用多进程调用 CUDA 模型，必须使用 'spawn' 模式
    mp.set_start_method('spawn', force=True)
    
    # 加载配置
    with open("llms/llm_configs.json", "r", encoding="utf-8") as f:
        llm_configs = json.load(f)
    with open("berts/bert_configs.json", "r", encoding="utf-8") as f:
        bert_configs = json.load(f)
    
    NUM_GPUS = 8
    llm_names = ["qwen3_5_9b"]
    bert_name = "bert-base-chinese"
   
    homograph_path = "homoGraphs/ToxiCN_homoGraph"
    data_path = "datas/ToxicloakCN/Toxic_key.json"
    output_path = "datas/processed_data/ToxicloakCN/"

    # homograph_path = "homoGraphs/ToxiCN_homoGraph"
    # data_path = "datas/ToxicloakCN/Toxic_key.json"
    # output_path = "datas/processed_data/ToxicloakCN/"

    # homograph_path = "homoGraphs/COLD_homoGraph"
    # data_path = "datas/COLD/COLD_key.json"
    # output_path = "datas/processed_data/COLD/"

    # homograph_path = "homoGraphs/COLD_shapeGraph"
    # data_path = "datas/COLD/COLD_shape_key.json"
    # output_path = "datas/processed_data/COLD/"

    # real world data test
    # homograph_path = "homoGraphs/ToxiCN_homoGraph"
    # data_path = "datas/Toxic_rest/Toxic_real_world.json"
    # output_path = "datas/processed_data/ToxicloakCN/"

    drop_edge = False
    drop_lexicon = False
    probs = [0.1, 0.3, 0.5]
    prob = 1.0
    os.makedirs(output_path, exist_ok=True) # 确保输出文件夹存在

    # for prob in probs:

    for llm_name in llm_names:
        print(f"========== 开始处理 LLM: {llm_name} ==========")
        start_time = time.time()
        
        # 1. 读取全局数据
        with open(data_path, "r", encoding="utf-8") as f:
            datas = json.load(f)
        
        # 2. 将数据切分为 8 块
        N = len(datas)
        chunk_size = math.ceil(N / NUM_GPUS)
        chunks = [datas[i : i + chunk_size] for i in range(0, N, chunk_size)]
        
        # 3. 准备进程池任务参数
        tasks = []
        for i, chunk in enumerate(chunks):
            # 保证即使块数小于 NUM_GPUS 也不会越界
            gpu_id = i % NUM_GPUS
            tasks.append((
                gpu_id, 
                chunk, 
                llm_name, 
                llm_configs[llm_name], 
                bert_configs[bert_name], 
                homograph_path,
                drop_edge,
                drop_lexicon,
                prob
            ))
        
        # 4. 启动多进程池
        print(f"分配完毕，共切分为 {len(chunks)} 块，启动多进程池...")
        with mp.Pool(processes=len(chunks)) as pool:
            # starmap 会阻塞直到所有进程完成，并按任务顺序返回结果列表
            results = pool.starmap(worker_process, tasks)
        
        print("\n\n所有进程处理完毕，开始合并与保存数据...")
        
        # 5. 合并各个进程的结果并保存
        # results 是一个列表，里面包含了每个进程返回的 chunk_results 字典
        if results:
            modes = results[0].keys()
            for mode in modes:
                if "llm-m" not in mode:
                    continue
                merged_data = []
                for res in results:
                    merged_data.extend(res[mode])
                
                # 拼接文件名
                dataset_name = f"ToxicloakCN_homoGraph_dn"
                if "llm" in mode:
                    dataset_name += f"_{llm_name}"
                dataset_name += f"_{mode}"
                output_file = os.path.join(output_path, f"{dataset_name}.json")
                
                with open(output_file, "w", encoding="utf-8") as _of:
                    json.dump(merged_data, _of, ensure_ascii=False, indent=4)
                print(f"✅ 成功保存合并后的文件: {output_file} (总数据条数: {len(merged_data)})")

        end_time = time.time()
        # print(f"Total time ({llm_name}): {(end_time - start_time):.2f} s\n")

