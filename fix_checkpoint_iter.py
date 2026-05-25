import argparse
import torch


def main(path: str, new_iter: int):
    ckpt = torch.load(path, weights_only=False)
    ckpt["iter"] = new_iter
    torch.save(ckpt, path)
    print(f"Updated {path} iter -> {new_iter}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="checkpoint file path")
    parser.add_argument("iter", type=int, help="new iter value")
    args = parser.parse_args()
    main(args.path, args.iter)
