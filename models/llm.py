import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM



class LLM_model():
    def __init__(self, llm_configs, device="cpu"):
        llm_path = llm_configs["path"]
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            use_cache=True,
            trust_remote_code=True,
        ).to(device)
        self.device = device
    
    def text_generation(self, text: str, max_new_tokens=64):
        inputs = self.tokenizer(
            text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            output_scores=True,
            return_dict_in_generate=True,
        )
        ans_begin = inputs["input_ids"].size(1)
        output_tokens = output.sequences[0][ans_begin:]
        if output_tokens[-1] == self.tokenizer.eos_token_id:
            output_tokens = [output_tokens[:-1]]
        ans = self.tokenizer.decode(output_tokens).strip()

        logits = output.scores[0][0]

        return ans, logits

    def get_self_p(self, text):
        input_ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=True)["input_ids"]
        N = len(input_ids[0])
        p = 0.
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.device))
            logits = outputs.logits[0]  # S, dict_size
            probs = torch.softmax(logits, dim=1)
            for i in range(N-1):
                p += torch.log(probs[i][input_ids[0][i+1]])
        return p

    def bayes(self, text, lexicons):
        ps = []
        for lexicon in lexicons:
            pos, word = lexicon["pos"], lexicon["lexicon"]
            ps.append((self.get_self_p(text), self.get_self_p(
                text[:pos] + word + text[pos + len(word):])))
        for i, p in enumerate(ps):
            if p[0] < p[1]:
                pos, word = lexicons[i]["pos"], lexicons[i]["lexicon"]
                text = text[:pos] + word + text[pos + len(word):]

        return text

    def multi_bayes(self, text, lexicons):
        replaced_pos_word = []
        while True:
            minn = (0, 0, 0, -1)
            for i, lexicon in enumerate(lexicons):
                if i in replaced_pos_word:
                    continue
                pos, word = lexicon["pos"], lexicon["lexicon"]
                p0, p1 = self.get_self_p(text), self.get_self_p(
                    text[:pos] + word + text[pos + len(word):])
                if p0 - p1 < minn[2]:
                    minn = (pos, word, p0-p1, i)
            if minn[3] == -1:
                break
            pos, word, _, i = minn
            text = text[:pos] + word + text[pos + len(word):]
            replaced_pos_word.append(i)
        return text
    
    def get_next_logits_from_tokens(self, tokens, max_new_tokens=1):
        inputs = {"input_ids": torch.tensor([tokens], dtype=torch.int64), "attention_mask": torch.ones(
            (1, len(tokens)), dtype=torch.int64)}
        output = self.model.generate(
            inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            output_scores=True,
            return_dict_in_generate=True,
        )
        logits = output.scores[0][0]
        return logits
    
    def compute_next_word_prob(self, text, word):
        logp = 0.
        text_tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(text_tokens) == 0:
            return 0.
        word_tokens = self.tokenizer.encode(word, add_special_tokens=False)

        for i, word_token in enumerate(word_tokens):
            logits = self.get_next_logits_from_tokens(text_tokens)
            probs = nn.functional.softmax(logits, dim=0)
            logp += torch.log(probs[word_token]).item()
            text_tokens.append(word_token)

        return logp / len(word_tokens)
    
    def pre_llm(self, text, lexicons):
        ps = []
        for lexicon in lexicons:
            pos, word = lexicon["pos"], lexicon["lexicon"]
            ps.append((self.compute_next_word_prob(
                text[:pos], text[pos:pos+len(word)]), self.compute_next_word_prob(text[:pos], word)))
        for i, p in enumerate(ps):
            if p[0] < p[1]:
                pos, word = lexicons[i]["pos"], lexicons[i]["lexicon"]
                text = text[:pos] + word + text[pos + len(word):]
        return text
    def pure_llm(self, text):
        instruct = "任务：针对中文敏感信息的错别字纠正。\n要求：需要严格保证输入句子和输出句子长度一致，只需要返回纠正后的句子，不要输出其他任何内容。\n输入：%s\n输出：" % (
            text)
        return self.text_generation(instruct, max_new_tokens=512)[0]
