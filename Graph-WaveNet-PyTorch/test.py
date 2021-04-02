import os
import util
import argparse
from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='./data', help='data path')
parser.add_argument('--adjdata', type=str, default='../data/adj_mx_geo_126.csv', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int,default=12, help='')
parser.add_argument('--nhid', type=int, default=64, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=126, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--checkpoint', type=str, default='save/best_model.pth', help='')
parser.add_argument('--plotheatmap',type=bool,default=False, help='')
args = parser.parse_args()

def main():
    device = torch.device(args.device)
    adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    adjinit = supports[0] if args.randomadj else None

    if args.aptonly:
        supports = None

    model = gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool,
        addaptadj=args.addaptadj, aptinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length,
        residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=args.nhid * 8,
        end_channels=args.nhid * 16)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    print('Model load successfully')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    truth = torch.Tensor(dataloader['y_test']).to(device)
    truth = truth.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    y_hat = torch.cat(outputs,dim=0)
    y_hat = y_hat[:truth.size(0),...]

    acc_mean, pred_all, truth_all = [], [], []
    for i in range(12):
        pred_i = scaler.inverse_transform(y_hat[:,:,i]).cpu().detach().numpy()
        truth_i = truth[:,:,i].cpu().detach().numpy()
        acc, accH, accN, accL = util.metric_acc(pred_i.reshape(-1), truth_i.reshape(-1))
        print(f'Horizon {(i+1):02d}, Acc: {acc:.4f}, AccH: {accH:.4f}, AccN: {accN:.4f}, AccL: {accL:.4f}')
        acc_mean.append(acc)
        pred_all.append(pred_i)
        truth_all.append(truth_i)

    log = 'Average acc: {:.4f}'
    print(log.format(np.mean(acc_mean)))
    output = {'prediction': pred_all, 'truth': truth_all}
    np.savez_compressed('./predicted_results.npz', **output)


if __name__ == "__main__":
    main()
