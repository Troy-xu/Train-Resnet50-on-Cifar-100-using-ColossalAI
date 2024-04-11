# Train-Resnet50-on-Cifar-100-using-ColossalAI

This assignment is only for educational experiment purpose on familiarize with ColossalAI.

Colossal-AI is an integrated large-scale deep learning system with efficient parallelization techniques. The system can accelerate model training on distributed systems with multiple GPUs by applying parallelization techniques. The system can also run on systems with only one GPU. In this assignment I am using colab which only has single GPU.

## Quick Start
This example provides a training script and an evaluation script. The training script provides an example of training ResNet50 on CIFAR100 dataset from scratch.

* Training Arguments

  * -p, --plugin: Plugin to use. Choices: torch_ddp, torch_ddp_fp16, low_level_zero. Defaults to torch_ddp.
  * -r, --resume: Resume from checkpoint file path. Defaults to -1, which means not resuming.
  * -c, --checkpoint: The folder to save checkpoints. Defaults to ./checkpoint.
  * -i, --interval: Epoch interval to save checkpoints. Defaults to 5. If set to 0, no checkpoint will be saved.
  * --target_acc: Target accuracy. Raise exception if not reached. Defaults to None.
* Eval Arguments

  * -e, --epoch: select the epoch to evaluate
  * -c, --checkpoint: the folder where checkpoints are found
  
### Set up in colab 
* Mount to google drive
```
from google.colab import drive
drive.mount('/content/drive')
```

* Install requirements
```
pip install -r requirements.txt
```
It will pop up a window to ask you to restart to use the runtime
* Locate to your working directory
```
cd "/content/drive/MyDrive/Colab Notebooks/NUS MCOMP Sem2/CS5260/asg6/Resnet50_Cifar100"
```
* Make sure current directory is correct
```
pwd
```

### Train
The folders will be created automatically. (--nproc_per_node n, where n is the number of GPU in your machine)
```
# train with torch DDP with fp32
!colossalai run --nproc_per_node 1 train.py -c ./ckpt-fp32

# train with torch DDP with mixed precision training
!colossalai run --nproc_per_node 1 train.py -c ./ckpt-fp16 -p torch_ddp_fp16

# train with low level zero
!colossalai run --nproc_per_node 1 train.py -c ./ckpt-low_level_zero -p low_level_zero
```
### Eval
```
# evaluate fp32 training
!python eval.py -c ./ckpt-fp32 -e 80

# evaluate fp16 mixed precision training
!python eval.py -c ./ckpt-fp16 -e 80

# evaluate low level zero training
!python eval.py -c ./ckpt-low_level_zero -e 80
```
### Colab logs
You can see the notebook logs directly from:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Troy-xu/Train-Resnet50-on-Cifar-100-using-ColossalAI/blob/main/Resnet50_Cifar100.ipynb)

