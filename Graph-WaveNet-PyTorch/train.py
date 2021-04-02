import os
import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/',help='data path')
parser.add_argument('--adjdata',type=str,default='../data/adj_mx_geo_126.csv',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=64,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=126,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='save/',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')

args = parser.parse_args()


def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    print(args)
    
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit)

    print("Start training...",flush=True)
    his_loss =[]
    val_time, train_time = [], []
    for i in range(1, args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss, train_mape, train_rmse = [], [], []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for idx, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
        t2 = time.time()
        train_time.append(t2-t1)
        
        #validation
        valid_loss, valid_mape, valid_rmse = [], [], []
        s1 = time.time()
        for idx, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        infer_time = time.time() - s1
        print(f'Epoch: {i:03d}, Inference Time: {infer_time:.4f} secs')
        val_time.append(infer_time)

        mtrain_loss, mtrain_mape, mtrain_rmse = np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)
        mvalid_loss, mvalid_mape, mvalid_rmse = np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse)
        
        his_loss.append(mvalid_loss)
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    best_epoch = np.argmin(his_loss)
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[best_epoch],4)), 'best model: ', best_epoch)
    torch.save(engine.model.state_dict(), args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[best_epoch],2)) + ".pth")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total time spent: {:.4f}".format(end_time-start_time))
