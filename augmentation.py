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
import cv2 as cv
import random

def rotate_bound_black_bg(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv.warpAffine(image, M, (nW, nH),borderValue=(0,0,0))
    # return cv2.warpAffine(image, M, (nW, nH))

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
    elif method == "Rotate":                 #翻转及旋转
        for i in [-6, -3, 3, 6]:
            newImage = rotate_bound_black_bg(input_pic, i)
            newH, newW = newImage.shape[:2]
            outputImage = newImage[newH // 2 - h // 2: newH // 2 + h // 2, newW // 2 - w // 2: newW // 2 + w // 2, :]
            output.append(outputImage)
            label.append(input_label)
        newImage1 = input_pic[:, ::-1, :]
        output.append(newImage1)
        label.append(input_label)
        for i in [-6, -3, 3, 6]:
            newImage = rotate_bound_black_bg(newImage1, i)
            newH, newW = newImage.shape[:2]
            outputImage = newImage[newH // 2 - h // 2: newH // 2 + h // 2, newW // 2 - w // 2: newW // 2 + w // 2, :]
            output.append(outputImage)
            label.append(input_label)
    elif method == "Scale":                  #缩放
        for i in [0.1, 0.2, 0.3]:
            newH, newW = int(h * (1 + i)), int(w * (1 + i))
            newImage = cv.resize(input_pic, (newH, newW))
            hO, wO = random.randint(0, newH - h), random.randint(0, newW - w)
            outputImage = newImage[hO:hO + h, wO:wO + w, :]
            output.append(outputImage)
            label.append(input_label)
            hO, wO = random.randint(0, newH - h), random.randint(0, newW - w)
            outputImage = newImage[hO:hO + h, wO:wO + w, :]
            output.append(outputImage)
            label.append(input_label)
            hO, wO = random.randint(0, newH - h), random.randint(0, newW - w)
            outputImage = newImage[hO:hO + h, wO:wO + w, :]
            output.append(outputImage)
            label.append(input_label)
    return output, label