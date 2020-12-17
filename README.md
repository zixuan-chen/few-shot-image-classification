# few-shot-image-classification
few shot image classification

### Experiment Result

| model | mini 1-shot | mini 5-shot | FC100 1-shot | FC100 5-shot |
| ----  | ----        | ----        | ----         | ----         |  
| ResNet(No aug) | 0.5328 + 0.0080| 0.6857 + 0.0070 | 0.3922 + 0.0071 | 0.5153 + 0.0070 | 

### train command
这里gpu需要根据自己显卡的情况来设置，一般设置为0

`cd meta-transfer-learning`
- meta_train(ResNet12, MiniImageNet, 5 shot, with augmentation)

`python main.py --phase meta_train --dataset MiniImageNet --shot 5 --model_type ResNet --augment 1 --pre_batch_size 128 --pre_lr 0.1 --gpu xxx`

- meta_eval(ResNet12, MiniImageNet, 5 shot, with augmentation)
`python main.py  --phase meta_train --dataset MiniImageNet --shot 5 --model_type ResNet --augment 1 --pre_batch_size 128 --pre_lr 0.1 --gpu xxx`

- meta_train(ResNet12, MiniImageNet, 1 shot, with augmentation)

`python main.py --phase meta_train --dataset MiniImageNet --shot 5 --model_type ResNet --augment 1 --pre_batch_size 128 --pre_lr 0.1  --gpu xxx`

- meta_eval(ResNet12, MiniImageNet, 1 shot, with augmentation)
`python main.py  --phase meta_train --dataset MiniImageNet --shot 5 --model_type ResNet --augment 1 --pre_batch_size 128 --pre_lr 0.1 --gpu xxx`

