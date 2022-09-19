import torch

def log_softmax(x):
    return torch.log(torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True))

def ce_loss(pred, y):
    loss = -y*log_softmax(pred)
    return 2*loss

def wce_loss(pred, y, beta, dev):
    # print(pred.requires_grad, y.requires_grad)
    # criterion = nn.CrossEntropyLoss()
    p = torch.unsqueeze(torch.argmax(pred, 1), 1)
    # print(p.shape, y.shape)
    ytrue = torch.cat((beta*(1-y), y), 1).cuda(dev)
    # print(pred.shape, ytrue.shape, beta)
    # print(pred[0, :, 0, :10], ytrue[0, :, 0, :10])
    # loss = criterion(pred, torch.squeeze(y).long())
    ce = ce_loss(pred, ytrue)
    # print(loss, loss.item(), loss.requires_grad)
    # print(ce, ce.item(), ce.requires_grad)
    return torch.mean(ce)