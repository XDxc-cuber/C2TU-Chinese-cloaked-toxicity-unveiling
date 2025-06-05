import json
import re

def sent_metric_cor(all_srcs, all_pres, all_trgs):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    change_num = 0
    for src, tgt_pred, tgt in zip(all_srcs, all_pres, all_trgs):
        if src != tgt_pred:
            change_num += 1
        if len(src) != len(tgt_pred):
            FN += 1
        elif src == tgt:
            if tgt == tgt_pred:
                TN += 1
            else:
                FP += 1
        else:
            if tgt == tgt_pred:
                TP += 1
            else:
                FN += 1
        total_num += 1
    acc = (TP + TN) / total_num
    precision = TP / change_num if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision + recall != 0 else 0
    print(
        f'Sentence Level correction: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')
    return acc, precision, recall, f1


def sent_metric_det(all_srcs, all_pres, all_trgs):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    change_num = 0
    for src, tgt_pred, tgt in zip(all_srcs, all_pres, all_trgs):
        if src != tgt_pred:
            change_num += 1
        total_num += 1
        if len(src) != len(tgt_pred):
            FN += 1
            continue
        
        src_tgt_tag = [1 if s == t else 0 for s,
                       t in zip(list(src), list(tgt))]
        src_tgt_pred_tag = [1 if s == t else 0 for s,
                            t in zip(list(src), list(tgt_pred))]

        if src == tgt:
            if src == tgt_pred:
                TN += 1
            else:
                FP += 1
        else:
            if src_tgt_tag == src_tgt_pred_tag:
                TP += 1
            else:
                FN += 1
        
    acc = (TP + TN) / total_num
    precision = TP / change_num if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision + recall != 0 else 0
    print(
        f'Sentence Level detection: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')
    return acc, precision, recall, f1


def char_metric_cor(all_srcs, all_pres, all_trgs):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    for src, tgt_pred, tgt in zip(all_srcs, all_pres, all_trgs):
        N = min(len(src), len(tgt_pred), len(tgt))
        # if len(tgt) != N or len(tgt_pred) != N:
        #     continue
        # assert len(tgt) == N and len(tgt_pred) == N

        for i in range(N):
            if src[i] == tgt[i]:
                if tgt_pred[i] == tgt[i]:
                    TN += 1
                else:
                    FP += 1
            else:
                if tgt_pred[i] == tgt[i]:
                    TP += 1
                else:
                    FN += 1
        total_num += N
    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision + recall != 0 else 0
    print(
        f'Character Level correction: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')
    return acc, precision, recall, f1


def char_metric_det(all_srcs, all_pres, all_trgs):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    for src, tgt_pred, tgt in zip(all_srcs, all_pres, all_trgs):
        N = min(len(src), len(tgt_pred), len(tgt))
        src, tgt_pred, tgt = src[:N], tgt_pred[:N], tgt[:N]
        src_tgt_tag = [0 if s == t else 1 for s,
                       t in zip(list(src), list(tgt))]
        src_tgt_pred_tag = [0 if s == t else 1 for s,
                            t in zip(list(src), list(tgt_pred))]
        # N = len(src_tgt_tag)
        # if len(src_tgt_pred_tag) != N:
        #     continue
        # assert len(src_tgt_pred_tag) == N
        
        for i in range(N):
            if src_tgt_tag[i] == 0:
                if src_tgt_pred_tag[i] == 0:
                    TN += 1
                else:
                    FP += 1
            else:
                if src_tgt_pred_tag[i] == 1:
                    TP += 1
                else:
                    FN += 1
        total_num += N
    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision + recall != 0 else 0
    print(
        f'Character Level detection: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')
    return acc, precision, recall, f1


def char_metric_cor_(all_srcs, all_pres, all_trgs):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    for src, tgt_pred, tgt in zip(all_srcs, all_pres, all_trgs):
        # N = len(src)
        # if len(tgt) != N or len(tgt_pred) != N:
        #     continue
        # assert len(tgt) == N and len(tgt_pred) == N
        N = min(len(src), len(tgt_pred), len(tgt))
        src, tgt_pred, tgt = src[:N], tgt_pred[:N], tgt[:N]

        for i in range(N):
            if src[i] == tgt[i]:
                if tgt_pred[i] == tgt[i]:
                    TN += 1
                else:
                    FP += 1
            else:
                if tgt_pred[i] == tgt[i]:
                    TP += 1
                else:
                    FN += 1
        total_num += N
    
    return TP/total_num, FP/total_num, FN/total_num, TN/total_num


def char_metric_det_(all_srcs, all_pres, all_trgs):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    for src, tgt_pred, tgt in zip(all_srcs, all_pres, all_trgs):
        N = min(len(src), len(tgt_pred), len(tgt))
        src, tgt_pred, tgt = src[:N], tgt_pred[:N], tgt[:N]
        
        src_tgt_tag = [0 if s == t else 1 for s,
                       t in zip(list(src), list(tgt))]
        src_tgt_pred_tag = [0 if s == t else 1 for s,
                            t in zip(list(src), list(tgt_pred))]
        # N = len(src_tgt_tag)
        # if len(src_tgt_pred_tag) != N:
        #     continue
        # assert len(src_tgt_pred_tag) == N
        
        for i in range(N):
            if src_tgt_tag[i] == 0:
                if src_tgt_pred_tag[i] == 0:
                    TN += 1
                else:
                    FP += 1
            else:
                if src_tgt_pred_tag[i] == 1:
                    TP += 1
                else:
                    FN += 1
        total_num += N
    return TP/total_num, FP/total_num, FN/total_num, TN/total_num


def read_datas(data_path):
    datas = []
    with open(data_path, "r", encoding="utf-8") as f:
        datas = json.loads(f.readlines()[0])
    return datas

def eval_denoisy(base_path, key_path, eval_path):
    base_datas, key_datas, eval_datas = read_datas(base_path), read_datas(key_path), read_datas(eval_path)
    base_texts, key_texts, eval_texts = [], [], []
    N = len(base_datas)
    for i in range(N):
        base_text, key_text, eval_text = base_datas[i]["text"], key_datas[i]["text"], eval_datas[i]["text"]
        if not (len(base_text) == len(key_text) and len(base_text) == len(eval_text)):
            base_text = base_text.strip("\n \'\"")
            key_text = key_text.strip("\n \'\"")
            eval_text = eval_text.strip("\n \'\"")
        
        base_texts.append(base_text)
        key_texts.append(key_text)
        eval_texts.append(eval_text)
        
    metric_list = []
    metric_list.append(sent_metric_det(key_texts, eval_texts, base_texts))
    metric_list.append(sent_metric_cor(key_texts, eval_texts, base_texts))
    metric_list.append(char_metric_det(key_texts, eval_texts, base_texts))
    metric_list.append(char_metric_cor(key_texts, eval_texts, base_texts))
    metric_list.append(char_metric_det_(key_texts, eval_texts, base_texts))
    metric_list.append(char_metric_cor_(key_texts, eval_texts, base_texts))
    return metric_list
    

if __name__ == "__main__":
    #base_path = "datas/COLD/COLD_base.json"
    #noise_path = "datas/COLD/COLD_key.json"
    base_path = "datas/ToxicloakCN/Toxic_base.json"
    noise_path = "datas/ToxicloakCN/Toxic_key.json"
    
    #eval_path = "datas/processed_data/COLD/"
    eval_path = "datas/processed_data/ToxicloakCN/"
    
    import os
    eval_datas = os.listdir(eval_path)
    
    result = {}
    for eval_data in eval_datas:
        path = eval_path + eval_data
        print(path)
        metric_list = eval_denoisy(base_path, noise_path, path)
        result[path] = []
        for metric in metric_list:
            for x in metric:
                result[path].append(x)
        print()
    
    with open("result.csv", "w") as f:
        f.write(
            "model,SD-A,SD-P,SD-R,SD-F,SC-A,SC-P,SC-R,SC-F,CD-A,CD-P,CD-R,CD-F,CC-A,CC-P,CC-R,CC-F,CD-TP,CD-FP,CD-FN,CD-TN,CC-TP,CC-FP,CC-FN,CC-TN,\n")
        for key, value in result.items():
            f.write(key)
            for x in value:
                f.write(",%.2f"%(x*100))
            f.write("\n")
        
    
