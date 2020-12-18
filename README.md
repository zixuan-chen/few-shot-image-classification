# Augmented Meta-Transfer Learning(A-MTL)
 for few shot image classification
## Datasets

Directly download processed images: [\[Download Page\]](https://mtl.yyliu.net/download/)

### 𝒎𝒊𝒏𝒊ImageNet

The 𝑚𝑖𝑛𝑖ImageNet dataset was proposed by [Vinyals et al.](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) for few-shot learning evaluation. Its complexity is high due to the use of ImageNet images but requires fewer resources and infrastructure than running on the full [ImageNet dataset](https://arxiv.org/pdf/1409.0575.pdf). In total, there are 100 classes with 600 samples of 84×84 color images per class. These 100 classes are divided into 64, 16, and 20 classes respectively for sampling tasks for meta-training, meta-validation, and meta-test. To generate this dataset from ImageNet, you may use the repository [𝑚𝑖𝑛𝑖ImageNet tools](https://github.com/y2l/mini-imagenet-tools).

### Fewshot-CIFAR100

Fewshot-CIFAR100 (FC100) is based on the popular object classification dataset CIFAR100. The splits were
proposed by [TADAM](https://arxiv.org/pdf/1805.10123.pdf). It offers a more challenging scenario with lower image resolution and more challenging meta-training/test splits that are separated according to object super-classes. It contains 100 object classes and each class has 600 samples of 32 × 32 color images. The 100 classes belong to 20 super-classes. Meta-training data are from 60 classes belonging to 12 super-classes. Meta-validation and meta-test sets contain 20 classes belonging to 4 super-classes, respectively.


## Performance
### MTL performance
| backbone | Aug method |mini 1-shot | mini 5-shot | FC100 1-shot | FC100 5-shot |
| ----  | ----       |----        | ----        | ----         | ----         |  
| ResNet12| None     | 0.5328 + 0.0080| 0.6857 + 0.0070 | 0.3922 + 0.0071 | 0.5153 + 0.0070 | 
|         | Transport| 0.5342 + 0.0085 | 0.6903 + 0.0070 | 0.4007+0.0079 | 0.5119 + 0.0071
| ResNet18| None     | 0.4632 + 0.0077| 0.6618 + 0.0069 | 0.3749 + 0.0078 | 0.5224 + 0.0071|
| |Transport | 0.4734 + 0.0074 | 0.6597 + 0.0069 | 0.3804 + 0.0069 | 0.5256+0.0069 |

## MAML
see README of `maml`
## Meta-Transfer Learning
To reproduce the result of experiment, do the following command

`cd /meta-transfer-learning`
### pre_train command
- pre_train(ResNet12, MiniImageNet)
  
`python main.py --phase pre_train --model_type ResNet --dataset MiniImageNet --dataset_dir ../data/mini-imagenet --pre_batch_size 100 --pre_lr 0.1 --gpu xxx`
- pre_train(ResNet18, MiniImageNet)
  
`python main.py --phase pre_train --model_type ResNet18 --dataset MiniImageNet --dataset_dir ../data/mini-imagenet --pre_batch_size 64 --pre_lr 0.01 --gpu xxx`


### meta_train command
- meta_train(ResNet12, MiniImageNet, 1 shot)

`python main.py --phase meta_train --dataset MiniImageNet --dataset_dir ../data/mini-imagenet/ --shot 1 --model_type ResNet --pre_batch_size 128 --pre_lr 0.1 --gpu xxx`

- meta_train(ResNet12, FC100, 1 shot)

`python main.py --phase meta_train --dataset FC100 --dataset_dir ../data/fc100/ --shot 1 --model_type ResNet  --pre_batch_size 128 --pre_lr 0.1  --gpu xxx`

### meta_eval command
- meta_eval(ResNet12, MiniImageNet, 5 shot)

`python main.py  --phase meta_train --dataset MiniImageNet --dataset_dir ../data/mini-imagenet/ --shot 5 --model_type ResNet  --pre_batch_size 128 --pre_lr 0.1 --gpu xxx`

- meta_eval(ResNet12, FC100, 5 shot)

`python main.py  --phase meta_train --dataset FC100 --dataset_dir ../data/fc100/ --shot 5 --model_type ResNet --pre_batch_size 128 --pre_lr 0.1 --gpu xxx`

- 需要注意的地方：
    1. `--gpu`需要根据自己显卡的情况来设置，一般设置为0
    2. 如果要跑Resnet18只需将`--model_type ResNet`改为`--model_type ResNet18`
    3. 如果要训练FC100数据集只需设置`--dataset FC100` `dataset_dir ../data/fc100`
    4. `--dataset_dir` 需要根据实际情况选择数据集的路径
    5. 一个模型的pre_train 大概需要4-5h, meta_train 大概需要1-2小时
    6. 对于我们的A-MTL方法，只需在meta_train 和meta_eval 阶段加入`--augment 1`即可