'''
Function: dataAugmentation
Input:
    - input_pic, 3-dim npArray
    - input_label, any type
    - method, currently support Translation ('Trans')
Output:
    - output, the list of output image, a list of 3-dim npArray
    - label, a list of labels
'''

import numpy as np

def dataAugmentation(input_pic, input_label, method = 'Trans'):
    h, w, c = input_pic.shape
    output, label = [input_pic], [input_label]
    if method == "Trans":                    #平移
        transH = [h // 10, h // 5]
        transW = [w // 10, w // 5]
        for i in transH:
            newImage = input_pic.copy()
            newImage[:h - i, :, :] = input_pic[i:, :, :]
            newImage[h - i:, :, :] = newImage[h - i - 1, :, :]
            output.append(newImage)
            label.append(input_label)
            newImage = input_pic.copy()
            newImage[i:, :, :] = input_pic[:h - i, :, :]
            newImage[:i, :, :] = newImage[i, :, :]
            output.append(newImage)
            label.append(input_label)
        for i in transW:
            newImage = input_pic.copy()
            newImage[:, :w - i, :] = input_pic[:, i:, :]
            for j in range(w - i, w):
                newImage[:, j, :] = newImage[:, w - i - 1, :]
            output.append(newImage)
            label.append(input_label)
            newImage = input_pic.copy()
            newImage[:, i:, :] = input_pic[:, :w - i, :]
            for j in range(i):
                newImage[:, j, :] = newImage[:, i, :]
            output.append(newImage)
            label.append(input_label)
    return output, label