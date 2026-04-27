import numpy as np
import torch
import torch.nn.functional as F

def cal_miou(pred, label, num_classes):
    '''
    输入参数：
    pred: 模型的预测输出。
    通常是一个形状为 (Batch_Size, Num_Classes, H, W) 的张量（Tensor），
    包含每个类别的置信度（Logits）。
    
    label: 真实标签（Ground Truth）。形状为 (Batch_Size, H, W) 的张量，
    每个像素值是对应类别的索引（$0, 1, 2, ...$）。
    
    num_classes: 整数，表示总类别数（包含背景）。
    
    输出结果：
    miou: 一个浮点数，代表除背景外所有类别的平均 IoU。
    ious: 一个列表，包含每个特定类别（不含背景）的 IoU 计算结果。
    当前计算miou忽略了背景，因为考虑背景会让miou虚高
    '''
    # pred = pred.cpu()
    # label = label.cpu()
    pred = pred.argmax(dim=1)
    # print('pred shape:', pred.shape)
    # print('label shape:', label.shape)
    ious = []
    for i in range(1, num_classes):
        pred_i = (pred == i)
        label_i = (label == i)
        
        TP = (pred_i & label_i).sum().item()
        TP_FP_FN = (pred_i | label_i).sum().item()
        
        if TP_FP_FN == 0:
            continue
        
        iou = TP / TP_FP_FN
        # print(f"class {i} IoU: {iou:.4f}")
        ious.append(iou)
    miou = np.mean(ious)
    return miou, ious

