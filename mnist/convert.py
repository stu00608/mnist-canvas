import torch
import argparse

from models import Net

IMG_SIZE = 28

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", required=True)
    args = parser.parse_args()

    model = Net()
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    dummy_input = torch.zeros((1, 1, IMG_SIZE, IMG_SIZE))
    torch.onnx.export(model, dummy_input,
                      'onnx_model.onnx', verbose=True)
