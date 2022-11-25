import torch
import argparse

from models import Net

IMG_SIZE = 28

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            description="A program to convert torch model with trained checkpoint to onnx format.")
    parser.add_argument("-c", "--ckpt", help="Path to .pt file.", required=True)
    parser.add_argument(
        '-o', '--out', help="Output .onnx file name (full path)", type=str)

    args = parser.parse_args()

    model = Net()
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    dummy_input = torch.zeros((1, 1, IMG_SIZE, IMG_SIZE))
    torch.onnx.export(model, dummy_input,
                      args.out, verbose=True)
