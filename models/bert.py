import torch
import torch.nn as nn
from transformers.models.bert import BertTokenizer, BertForMaskedLM



class Bert_mask_model():
    def __init__(self, model_path,  device="cuda"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def compute_p_mask(self, text, pos, lexicon):
        p = 0.
        new_pos = len(self.tokenizer.encode(
            text[:pos], add_special_tokens=False))
        for i in range(len(lexicon)):
            new_text = text[:pos+i] + \
                self.tokenizer.mask_token + text[pos+i+1:]
            tokens = self.tokenizer(
                new_text, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                output = self.model(**tokens)
                logits = output.logits[0]
            prob = torch.nn.functional.softmax(
                logits[new_pos+i+1], dim=0)[self.tokenizer.encode(text[pos+i])[1]]
            p += torch.log(prob).item()
        return p / len(lexicon)

    def getReplacedSentence(self, text: str, lexicons: list):
        ps = []
        for lexicon in lexicons:
            pos, word = lexicon["pos"], lexicon["lexicon"]
            replaced_text = text[:pos] + lexicon["lexicon"] + \
                text[pos + len(lexicon["lexicon"]):]
            ps.append((self.compute_p_mask(text, pos, word),
                        self.compute_p_mask(replaced_text, pos, word)))

        for index, p in enumerate(ps):
            if p[0] < p[1]:
                text = text[:lexicons[index]["pos"]] + lexicons[index]["lexicon"] + \
                    text[lexicons[index]["pos"] +
                         len(lexicons[index]["lexicon"]):]
        return text

    def multiGetReplacedSentence(self, text: str, lexicons: list):
        replaced_pos_word = []
        while True:
            minn = (0, 0, 0, -1)
            for i, lexicon in enumerate(lexicons):
                if i in replaced_pos_word:
                    continue
                pos, word = lexicon["pos"], lexicon["lexicon"]
                p0, p1 = self.compute_p_mask(text, pos, text[pos:pos+len(word)]), self.compute_p_mask(
                    text[:pos] + word + text[pos + len(word):], pos, word)
                if p0 - p1 < minn[2]:
                    minn = (pos, word, p0-p1, i)
            if minn[3] == -1:
                break
            pos, word, _, i = minn
            text = text[:pos] + word + text[pos + len(word):]
            replaced_pos_word.append(i)

        return text
