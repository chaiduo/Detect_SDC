import json

INPUT_JSONL = "/data1/home/dataset_share/cd_data/Qwen2.5-VL-7B/EarthVQA/final/detect_EarthVQA_Qwen_with_sem_project.jsonl"
OUTPUT_JSONL = "/data1/home/dataset_share/cd_data/Qwen2.5-VL-7B/EarthVQA/final/detect_EarthVQA_Qwen_with_sem_project_id_mod4_01.jsonl"

def main():
    count = 0
    with open(INPUT_JSONL, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_JSONL, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            item_id = data.get('id')
            if item_id is not None and item_id % 4 in (0, 1):
                fout.write(line + '\n')
                count += 1
    print(f"Done. Wrote {count} records to {OUTPUT_JSONL}")

if __name__ == '__main__':
    main()
