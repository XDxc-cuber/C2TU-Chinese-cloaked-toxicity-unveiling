import json
from pathlib import Path

GENERAL_PATH = Path("datas/processed_data/ToxicloakCN/ToxicloakCN_general_homoGraph_dn_baichuan7_llm-m.json")
ALL_PATH = Path("datas/processed_data/ToxicloakCN/ToxicloakCN_all_homoGraph_dn_baichuan7_llm-m.json")


def main():
    with GENERAL_PATH.open("r", encoding="utf-8") as f:
        general_data = json.load(f)
    with ALL_PATH.open("r", encoding="utf-8") as f:
        all_data = json.load(f)

    print(f"general samples: {len(general_data)}")
    print(f"all samples: {len(all_data)}")

    if len(general_data) != len(all_data):
        print("same length: False")
    else:
        print("same length: True")

    same_count = 0
    diff_count = 0
    first_diff = None

    for idx, (general_item, all_item) in enumerate(zip(general_data, all_data)):
        if general_item == all_item:
            same_count += 1
        else:
            diff_count += 1
            if first_diff is None:
                first_diff = (idx, general_item, all_item)

    diff_count += abs(len(general_data) - len(all_data))

    print(f"same samples: {same_count}")
    print(f"different samples: {diff_count}")
    print(f"all samples exactly equal: {diff_count == 0}")

    if first_diff is not None:
        idx, general_item, all_item = first_diff
        print(f"first different sample index: {idx}")
        print("general item:")
        print(json.dumps(general_item, ensure_ascii=False, indent=2))
        print("all item:")
        print(json.dumps(all_item, ensure_ascii=False, indent=2))

        general_text = general_item.get("text") if isinstance(general_item, dict) else None
        all_text = all_item.get("text") if isinstance(all_item, dict) else None
        if general_text != all_text:
            print("first different text:")
            print(f"general text: {general_text}")
            print(f"all text: {all_text}")


if __name__ == "__main__":
    main()
