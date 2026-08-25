import json
import sys
sys.path.append(".")
from homoGraphs.shapeGraph import ShapeGraph

paths = {
    "COLD": {
        "path": "datas/COLD/",
        "base": "datas/COLD/COLD_base.json",
        "lexicons": "datas/COLD/COLD_lexicons.json"
    },
    "ToxiCN": {
        "path": "datas/ToxicloakCN/",
        "base": "datas/ToxicloakCN/Toxic_base.json",
        "lexicons": "datas/ToxicloakCN/Toxic_lexicons.json"
    }
}
dataset = "ToxiCN"

lexicons = []
with open(paths[dataset]["lexicons"], "r", encoding="utf-8") as f:
    lexicons = json.loads("".join(f.readlines()))
base_datas = []
with open(paths[dataset]["base"], "r", encoding="utf-8") as f:
    base_datas = json.loads("".join(f.readlines()))


import random

random.seed(2025)
shapeGraph = ShapeGraph(f"homoGraphs/{dataset}_shapeGraph")

def get_neighbor(char, shapeGraph):
    if char not in shapeGraph.char2id:
        return [char]
    nbs = shapeGraph.A[shapeGraph.char2id[char]]
    nb_chars = []
    for i in range(len(nbs)):
        if nbs[i]:
            nb_chars.append(shapeGraph.id2char[i])
    return nb_chars


key_datas = []
for i, data in enumerate(base_datas):
    print(f"{i} / {len(base_datas)}", end="\r")
    text = data["text"]
    new_text = text[:]
    for lexicon in lexicons:
        if lexicon in text:
            lexicon_list = list(lexicon)
            for i, char in enumerate(lexicon_list):
                new_char = char
                new_chars = get_neighbor(char, shapeGraph)
                if len(new_chars):
                    new_char = random.choice(new_chars)
                lexicon_list[i] = new_char
            new_lexicon = "".join(lexicon_list)
            new_text = new_text.replace(lexicon, new_lexicon)
    key_datas.append({"text": new_text, "label": data["label"]})
print()
with open(paths[dataset]["path"] + f"{dataset}_shape_key.json", "x", encoding="utf-8") as f:
    f.write(json.dumps(key_datas, ensure_ascii=False))





