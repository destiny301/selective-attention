import torch

def log_softmax(x):
    return torch.log(torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True))

def ce_loss(pred, y):
    loss = -y*log_softmax(pred)
    return 2*loss

def wce_loss(pred, y, beta, dev):
    p = torch.unsqueeze(torch.argmax(pred, 1), 1)
    ytrue = torch.cat((beta*(1-y), y), 1).cuda(dev)
    ce = ce_loss(pred, ytrue)
    return torch.mean(ce)