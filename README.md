# few-shot-image-classification
few shot image classification

### Experiment Result

| model | Aug method |mini 1-shot | mini 5-shot | FC100 1-shot | FC100 5-shot |
| ----  | ----       |----        | ----        | ----         | ----         |  
| ResNet12| None     | 0.5328 + 0.0080| 0.6857 + 0.0070 | 0.3922 + 0.0071 | 0.5153 + 0.0070 | 
|         | Transport| 0.5342 + 0.0085 | 0.6903 + 0.0070 |
| ResNet18| None     | 0.4632 + 0.0077| 0.6618 + 0.0069 | 

### train command
`cd meta-transfer-learning`
- meta_train(ResNet12, MiniImageNet, 1 shot, with augmentation)

`python main.py --phase meta_train --dataset MiniImageNet --dataset_dir ../data/mini-imagenet/ --shot 1 --model_type ResNet --augment 1 --pre_batch_size 128 --pre_lr 0.1 --gpu xxx`

- meta_eval(ResNet12, MiniImageNet, 5 shot, with augmentation)

`python main.py  --phase meta_train --dataset MiniImageNet --dataset_dir ../data/mini-imagenet/ --shot 5 --model_type ResNet --augment 1 --pre_batch_size 128 --pre_lr 0.1 --gpu xxx`

- meta_train(ResNet12, FC100, 1 shot, with augmentation)

`python main.py --phase meta_train --dataset FC100 --dataset_dir ../data/fc100/ --shot 1 --model_type ResNet --augment 1 --pre_batch_size 128 --pre_lr 0.1  --gpu xxx`

- meta_eval(ResNet12, FC100, 5 shot, with augmentation)

`python main.py  --phase meta_train --dataset FC100 --dataset_dir ../data/fc100/ --shot 5 --model_type ResNet --augment 1 --pre_batch_size 128 --pre_lr 0.1 --gpu xxx`

- 需要注意的地方：
    1. `--gpu`需要根据自己显卡的情况来设置，一般设置为0
    2. `--dataset_dir` 需要根据实际情况选择数据集的路径
