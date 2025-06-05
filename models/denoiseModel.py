from models.llm import LLM_model
from models.bert import Bert_mask_model
from models.lexicon_detector import DetectorModel



class DenoiseModel():
    def __init__(self, llm_model: LLM_model, bert_model: Bert_mask_model, detector: DetectorModel):
        self.llm_model = llm_model
        self.bert_model = bert_model
        self.detector = detector
        self.modes = ["naive", 
                      "bert-m", "bert", 
                      "llm-m", "llm", "pre-llm", "pure-llm"]

    # all replace
    def naive_denoise(self, text, lexicons):
        for pos_lexicon in lexicons:
            pos, lexicon = pos_lexicon["pos"], pos_lexicon["lexicon"]
            text = text[:pos] + lexicon + text[pos+len(lexicon):]
        return text

    def denoise(self, text: str, mode="llm-m"):
        lexicons = self.detector.detectLexicons(text) # [{"pos": i, "lexicon": lexicon}]
        new_text = None
        
        if len(lexicons) == 0:
            return text
        
        if mode == "naive":
            new_text = self.naive_denoise(text, lexicons)
        elif mode == "bert-m":
            new_text = self.bert_model.multiGetReplacedSentence(text, lexicons)
        elif mode == "bert":
            new_text = self.bert_model.getReplacedSentence(text, lexicons)
        elif mode == "llm-m":
            new_text = self.llm_model.multi_bayes(text, lexicons)
        elif mode == "llm":
            new_text = self.llm_model.bayes(text, lexicons)
        elif mode == "pre-llm":
            new_text = self.llm_model.pre_llm(text, lexicons)
        elif mode == "pure-llm":
            new_text = self.llm_model.pure_llm(text)
        else:
            raise ValueError(mode)

        return new_text


if __name__ == "__main__":
    pass
