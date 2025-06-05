from openai import OpenAI

client = OpenAI(api_key="[api-key]",
                base_url="https://api.deepseek.com")


def api(text):
    prompt = """
    任务：
    针对中文敏感信息中带掩盖的错别字纠正。
    要求：
    1. 需要严格保证输入句子和输出句子长度一致；
    2. 只需要返回纠正后的句子，不要输出任何其他内容。
    """
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        stream=False
    )

    return response.choices[0].message.content


import json

if __name__ == "__main__":
    # data_path = "datas/ToxicloakCN/Toxic_key.json"
    # output_path = "datas/processed_data/ToxicloakCN/"

    data_path = "datas/COLD/COLD_key.json"
    output_path = "datas/processed_data/COLD/"

    

    with open(data_path, "r", encoding="utf-8") as f:
        datas = json.loads("".join(f.readlines()))
        N = len(datas)
        
        dataset_name = "deepseek-r1"
        output_file = output_path + dataset_name + ".json"
        new_datas = []
        for i, data in enumerate(datas):
            # print(" %d / %d" % (i, N), end="\r")
            text, label = data["text"].strip(" \n\'\""), data["label"]
            new_text = api(text)
            print(new_text)
            new_datas.append({"text": new_text, "label": label})
            if i == 1:
                break
        # with open(output_file, "x", encoding="utf-8") as _of:
        #     _of.write(json.dumps(new_datas, ensure_ascii=False))
        # print("\nSaved denoised file %s" % output_file)

