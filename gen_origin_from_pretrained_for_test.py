import argparse
import utils
import torch

def main(args):
    module = utils.EfficientNetModule(args.efficientnet_repo)
    EfficientNet = module.EfficientNet()
    pretrained = EfficientNet.from_pretrained(args.efficientnet_kind)
    state_dict = pretrained.state_dict()
    torch.save(state_dict, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genrate Pretrained Model")
    parser.add_argument("--efficientnet-kind", type=str, default="efficientnet-b0",
                        help="efficientnet kind (ex: efficientnet-b0, effcientnet-b1)")

    parser.add_argument("--efficientnet-repo", type=str, default="yet",
                        help="efficientnet kind (ex: yet(Yet Another EfficientNet-Pytorch), ept(Efficientnet-PyTorch))")

    parser.add_argument("--save-path", type=str, default=None,
                        help="save path")

    args = parser.parse_args()
    main(args)


