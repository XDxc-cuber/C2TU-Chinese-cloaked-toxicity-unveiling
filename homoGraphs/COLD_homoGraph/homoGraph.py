import numpy as np
import json



class HomoGraph:
    def __init__(self, path):
        self.A = np.load(path + "/homoGraphA.npy")
        self.char2id = []
        with open(path + "/char2id.json", "r") as f:
            self.char2id = json.loads(f.readlines()[0])
        self.id2char = []
        with open(path + "/id2char.json", "r") as f:
            self.id2char = json.loads(f.readlines()[0])
        self.lexicons = []
        with open(path + "/new_lexicon.json", "r") as f:
            self.lexicons = json.loads("".join(f.readlines()))
            
    
    def neighborsHas(self, char1, char2):
        if not char1 in self.char2id or not char2 in self.char2id:
            return 0
        id1, id2 = self.char2id[char1], self.char2id[char2]
        return self.A[id1][id2]
        
        
