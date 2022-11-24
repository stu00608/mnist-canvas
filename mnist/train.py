import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torchsummary import summary
from torchvision import datasets, transforms
from tqdm import tqdm

from models import Net
from utils import file_choices


def train(model, device, dataloader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    print(f'Epoch: {epoch}')
    tepoch = tqdm(dataloader, total=int(len(dataloader)))
    for data, target in tepoch:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Feed forward
        output = model(data)

        # Accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        # Loss
        loss = F.cross_entropy(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()

        # Step optimizer
        optimizer.step()

        tepoch.set_postfix(loss=loss.item())

    train_acc = correct / len(dataloader.dataset)
    train_loss = loss.item()

    print('Train Epoch: {} \t Loss: {:.4f} Accuracy: {:.4f}'.format(
        str(epoch+1), train_loss, train_acc))

    return train_acc, train_loss


def test(model, device, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(dataloader.dataset)
    test_loss /= len(dataloader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * test_acc))

    return test_acc, test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=lambda s: file_choices(("yaml"), s), default="config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    os.makedirs(config["data_path"], exist_ok=True)
    os.makedirs(config["ckpt_path"], exist_ok=True)

    # Set static random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Setting default device to cuda gpu.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
    ])

    train_dataset = datasets.MNIST(
        config["data_path"],
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.MNIST(
        config["data_path"],
        train=False,
        transform=test_transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())
    summary(model, (1, 28, 28))

    best_acc = 0.
    for epoch in range(config["epochs"]):
        train_acc, train_loss = train(
            model, device, train_dataloader, optimizer, epoch+1)
        test_acc, test_loss = test(model, device, test_dataloader)

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_path = os.path.join(
                config["ckpt_path"], f"e_{epoch+1}_acc_{test_acc}.pt")
            torch.save(model.state_dict(), ckpt_path)
