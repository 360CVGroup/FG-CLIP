import torch

def calculate_topk_accuracy(logits, labels, topk=(1, 5)):
    """
    计算图像分类的top1和top5准确率。

    Args:
        logits (torch.Tensor): 模型的输出logits,形状为 [batch_size, num_classes]。
        labels (torch.Tensor): 真实标签，形状为 [batch_size]。
        topk (tuple): 需要计算的topk准确率,默认为 (1, 5)。

    Returns:
        tuple: 包含top1和top5准确率的元组。
    """
    # 确保输入是torch.Tensor类型
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError("logits and labels must be torch.Tensor")
    
    # 确保logits和labels的设备一致
    if logits.device != labels.device:
        labels = labels.to(logits.device)
    
    # 计算topk准确率
    maxk = max(topk)
    batch_size = labels.size(0)
    
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size).item())
    
    return tuple(res)