import importlib.util
import os
import sys


def _load_earth_train_module():
    current_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(current_dir, "..", "EarthVQA", "train_mapping_model.py"))
    spec = importlib.util.spec_from_file_location("earth_train_mapping_model", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_earth_train = _load_earth_train_module()
LayerAwareResidualMLP = _earth_train.LayerAwareResidualMLP
train_model = _earth_train.train_model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="/data1/home/dataset_share/cd_data/llava-v1.5-7B/VQAv2/final/attn_proj_mapping_64_project.jsonl")
    parser.add_argument("--save_best_path", type=str, default="./model/best_mapping_model.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train_model(
        jsonl_path=args.jsonl_path,
        save_best_path=args.save_best_path,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
