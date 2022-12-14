# MNIST model Training

- Train the model to be deployed in web page.
- Convert the model to onnx format.

## Installation

- Need pytorch, PyYAML, tqdm to run scripts in this folder.

```
conda create -n <env_name> python=3.8
conda activate <env_name>
```

```
# According to https://pytorch.org/get-started/previous-versions/#v180
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# Or cpu only
conda install pytorch==1.8.0 torchvision==0.9.0 cpuonly -c pytorch
```

```
pip install -r requirements.txt
```

## Training

- It will store ckpts for each epoch.

```
python train.py
```

## Convert

- Copy the path of ckpt to convert to onnx.

```
# Configure filename to match yours.
python convert.py -c <checkpoint_filename.pt> -o <output_filename.onnx>
```
