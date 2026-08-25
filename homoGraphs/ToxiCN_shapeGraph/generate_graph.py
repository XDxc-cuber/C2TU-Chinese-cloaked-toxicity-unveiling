import json
import numpy as np

char2id, id2char = {}, []
with open("char2id.json", "r", encoding="utf-8") as f:
    char2id = json.loads(f.readlines()[0])
with open("id2char.json", "r", encoding="utf-8") as f:
    id2char = json.loads(f.readlines()[0])

def add_node(char2id, id2char, char):
    if char in char2id:
        return
    char2id[char] = len(id2char)
    id2char.append(char)
    return

added_chars = set()
with open("../similar_shape_relationship.txt", "r", encoding="utf-8") as f:
    relationships = f.readlines()
    for line in relationships:
        chars = line.strip().split(" ")
        if len(chars) == 1:
            continue
        for char in chars[1]:
            # 有一个字在字典中，就把两个节点都更新
            if char in char2id or chars[0] in char2id:
                added_chars.add(char)
                added_chars.add(chars[0])

for char in added_chars:
    add_node(char2id, id2char, char)

N = len(id2char)
A = np.zeros((N, N), dtype=bool)
for i in range(N):
    A[i, i] = 1

with open("../similar_shape_relationship.txt", "r", encoding="utf-8") as f:
    relationships = f.readlines()
    for line in relationships:
        chars = line.strip().split(" ")
        if len(chars) == 1:
            continue
        for char in chars[1]:
            if char in char2id and chars[0] in char2id:
                A[char2id[chars[0]]][char2id[char]] = A[char2id[char]][char2id[chars[0]]] = 1

print(N, A.sum()) # 5170 18370
np.save("shapeGraphA.npy", A)
with open("new_char2id.json", "x", encoding="utf-8") as f:
    f.write(json.dumps(char2id, ensure_ascii=False))
with open("new_id2char.json", "x", encoding="utf-8") as f:
    f.write(json.dumps(id2char, ensure_ascii=False))





