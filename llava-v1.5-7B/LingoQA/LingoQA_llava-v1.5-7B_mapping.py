import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.extend([parent_dir, grandparent_dir])

from llava_common import generate_answer_from_path, load_llava_model, set_seed
from utils.profiler import Profiler


def evaluate(
    model_path,
    val_file,
    data_dir,
    save_dir,
    device,
    max_new_tokens,
    model_base=None,
):
    tokenizer, model, image_processor, model_name = load_llava_model(model_path, device, model_base)

    prof = Profiler(model, proj_dim=64, seed=1234)
    prof.register()

    df = pd.read_parquet(val_file)

    os.makedirs(save_dir, exist_ok=True)

    sample_id = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = row["question"]
        images = row["images"]
        num_images = min(5, len(images))

        for i in range(num_images):
            img_path = images[i]
            image_path = os.path.join(data_dir, img_path)

            try:
                _ = generate_answer_from_path(
                    question=question,
                    image_path=image_path,
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                )

                prof.save_attn_proj_interlayer_jsonl(
                    os.path.join(save_dir, "attn_proj_interlayer.jsonl")
                )
            except Exception as e:
                print(f"[Warning] Failed on image: {image_path}")
                print(f"[Warning] Error: {e}")

            prof.reset(clear_stats=True)
            sample_id += 1

    prof.unregister()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--val_file", type=str, default="/data0/home/lc/cd/predict_error/LingoQA-main/data/val/val.parquet")
    parser.add_argument("--data_dir", type=str, default="/data0/home/lc/cd/predict_error/LingoQA-main/data/val/")
    parser.add_argument("--save_dir", type=str, default="/data1/home/dataset_share/cd_data/llava-v1.5-7B/LingoQA")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    set_seed(42)

    device = torch.device(args.device)

    evaluate(
        model_path=args.model_path,
        model_base=args.model_base,
        val_file=args.val_file,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
