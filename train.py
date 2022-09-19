import os
import argparse
from sklearn.metrics import confusion_matrix, precision_recall_curve, mean_absolute_error
import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader
import numpy as np
from torch import optim

from data.dsloader import Data
from model.unet import UNet, ULite
from util import wce_loss

def main(args):
    print(args)

    # load data
    print('============================loading data============================')
    root = os.path.join('../datasets', args.data) # dataset path
    dataset_tr = Data(root, args, 'train')
    dataset_te = Data(root, args, 'val')
    train_loader = DataLoader(dataset_tr, args.batchsz, shuffle=True)
    test_loader = DataLoader(dataset_te, args.batchsz, shuffle=True)

    # check cuda
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    print('training device:', device)

    # build model
    num_ch = 3 if args.channel == 'rgb' else 1
    num_cls = 1 if args.loss == 'bce' else 2
    model = UNet(n_channels=num_ch, n_classes=num_cls, bilinear=True) if args.lite == False else ULite(n_channels=num_ch, n_classes=num_cls, bilinear=True)
    model = model.to(device)
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    criterion = nn.BCELoss() if args.loss == 'bce' else nn.CrossEntropyLoss()

    # results saving path
    resultfolder = os.path.join('./result', args.data)
    if not os.path.exists(resultfolder):
        os.mkdir(resultfolder)

    modelfolder = os.path.join(resultfolder, 'checkpoint')
    if not os.path.exists(modelfolder):
        os.mkdir(modelfolder)
    aug = 'aug' if args.aug else 'ori'
    netname = 'ulite' if args.lite else 'unet'
    modelpath = os.path.join(modelfolder, str(args.imgsz)+'_'+args.channel+'_'+args.ds_method+'_'+aug+'_'+args.loss+'_'+netname+'_best.pth')
    final_modelpath = os.path.join(modelfolder, str(args.imgsz)+'_'+args.channel+'_'+args.ds_method+'_'+aug+'_'+args.loss+'_'+netname+'_final.pth')

    # train and validate
    print('============================Training============================')
    # model.load_state_dict(torch.load(modelpath))

    train_loss, test_loss = 0.0, 0.0
    tn, fp, fn, tp = 0.0, 0.0, 0.0, 0.0
    cm = np.zeros((2, 2))
    best_score = 0.0

    for epoch in range(args.epoch):
        # train model
        for xtr, ytr in train_loader:

            xtr, ytr = xtr.to(device), ytr.to(device)
            optimizer.zero_grad()
            ptr = model(xtr)

            if args.loss == 'wce':
                ltr = wce_loss(ptr, ytr, args.beta, device)
            elif args.loss == 'ce':
                ltr = criterion(ptr, torch.squeeze(ytr).long())
            else:
                ltr = criterion(ptr, ytr)
            ltr.backward()
            optimizer.step()

            train_loss += ltr.item()
        
        # evaluate model
        pred = np.zeros((0,1,args.imgsz,args.imgsz))
        gt = np.zeros((0,1,args.imgsz,args.imgsz))
        with torch.no_grad():
            for xte, yte in test_loader:
                xte, yte = xte.to(device), yte.to(device)
                pte = model(xte)

                if args.loss == 'wce':
                    lte = wce_loss(pte, yte, args.beta, device)
                elif args.loss == 'ce':
                    lte = criterion(pte, torch.squeeze(yte).long())
                else:
                    lte = criterion(pte, yte)
                test_loss += lte.item()
                
                if args.loss != 'bce':
                    pte = torch.unsqueeze(torch.argmax(pte, 1), 1)
                pte = pte.cpu().numpy()
                yte = yte.cpu().numpy()
                # print(pte.shape, yte.shape)
                yte[yte>=0.5] = 1
                yte[yte<0.5] = 0

                pred = np.append(pred, pte, axis=0)
                gt = np.append(gt, yte, axis=0)

                pte[pte>=0.5] = 1
                pte[pte<0.5] = 0
                cm += confusion_matrix(yte.astype(np.int32).flatten(), pte.flatten())

        pred = pred.flatten()
        gt = gt.flatten()
        precision, recall, threshold = precision_recall_curve(gt, pred)
        f_scores = 1.3*recall*precision/(recall+0.3*precision)

        mae = mean_absolute_error(gt, pred)
        tn, fp, fn, tp = cm.ravel()
        pr = tp/(tp+fp)
        rc = tp/(tp+fn)

        print('epoch', epoch+1, '\ttrain loss:', "{:.4f}".format(train_loss/len(train_loader)), '\ttest loss', "{:.4f}".format(test_loss/len(test_loader)), '\tMAE:', "{:.4f}".format(mae), '\tmaxf:', "{:.4f}".format(np.max(f_scores)),
        '\tIoU:', "{:.4f}".format(tp/(tp+fn+fp)), '\tTPR:', "{:.4f}".format(tp/(tp+fn)), '\tOverhead:', "{:.4f}".format(fp/(tp+fn)))

        if np.max(f_scores) > best_score:
            best_score = np.max(f_scores)
            best_mae, best_maxf, best_IoU, best_tpr, best_overhead = mae, np.max(f_scores), tp/(tp+fn+fp), tp/(tp+fn), fp/(tp+fn)
            torch.save(model.state_dict(), modelpath)
        torch.save(model.state_dict(), final_modelpath)
        cm = np.zeros((2, 2))
        train_loss = 0.0
        test_loss = 0.0
    
    print('============================Training Done!============================')
    print('final result:\n', '\tMAE:', "{:.4f}".format(best_mae), '\tmaxf:', "{:.4f}".format(best_maxf),
        '\tIoU:', "{:.4f}".format(best_IoU), '\tTPR:', "{:.4f}".format(best_tpr), '\tOverhead:', "{:.4f}".format(best_overhead))



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, help='which dataset to use(bdd100k/coco/MSRA10K)', default='MSRA10K')
    argparser.add_argument('--channel', type=str, help='rgb/grayscale', default='gray')
    argparser.add_argument('--imgsz', type=int, nargs='+', help='image size(for mots, need two values for height and width, eg. 320*180)', default=640)
    argparser.add_argument('--padsz', type=int, help='(optional)if pad to some image size first and then downsample, then use', default=512) # then downsample ratio would be padsz/imgsz
    argparser.add_argument('--ds_method', type=str, help='downsample method(max/mean/bilinear)', default='max')
    argparser.add_argument('--aug', action='store_true', help='data augmentation or not(ori/aug)')
    argparser.add_argument('--norm', action='store_true', help='normalize or not')

    argparser.add_argument('--dev', type=str, help='cuda device', default='cuda:1')
    argparser.add_argument('--epoch', type=int, help='number of training epochs', default=100)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparser.add_argument('--lr_slot', type=int, help='learning rate change point(related to batch size)', default=2000)
    argparser.add_argument('--batchsz', type=int, help='batch size(12/15/32/64/128)', default=128)
    argparser.add_argument('--loss', type=str, help='loss function(bce/ce/wce)', default='wce')
    argparser.add_argument('--beta', type=float, help='fn/fp ratio', default=0.001)

    # argparser.add_argument('--dir', type=str, help='model saving directory', default='mots320')
    argparser.add_argument('--lite', action='store_true', help='use simplified UNet or not')

    args = argparser.parse_args()
    main(args)
