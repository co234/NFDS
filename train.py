from ast import parse
import numpy as np
import argparse
import torch
import torch.nn as nn
import pickle
from utils import fairness_measure,load_data_generator
import pandas as pd
import torchvision.models as models
from filter_function import testing, zero_shot_predictor_loss,training_step,test_step


parser = argparse.ArgumentParser(description='fair-selection')
#Network
parser.add_argument('--batch_size', default=320, type=int, help='batch_size')
parser.add_argument('--dataset', type=str, choices=['lfwa_w', 'celeba'], default='lfwa_w')
parser.add_argument('--c', default=0.2, type=float)
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--gamma', default=0.3, type=float)
parser.add_argument('--train_ratio',default = 0.1,type=float)
parser.add_argument('--c_num', default=0.01, type=float)
parser.add_argument('--seed', type=int, default=828) 
parser.add_argument('--num', default=150, type=float, help='tau')
parser.add_argument("--num_class", default=2, type=int)
parser.add_argument('--epoch', default=15, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--opt_eps', default=0.001, type=float, help='tau')
parser.add_argument('--hidden_size', default=15, type=int)
parser.add_argument('--num_iter', default=15, type=int)
parser.add_argument('--min_epochs',default=1, type=int)
parser.add_argument('--max_epochs', default=100, type=int)
parser.add_argument('--trainset_data_aug',default=False, type=bool)
parser.add_argument('--valset_data_aug',default=True,type=bool)
parser.add_argument('--retrain_validation',default="retrain",type=str, choices = ["retrain","skip"])
parser.add_argument('--selection',default="sampling", type=str, choices=["sampling","no_sampling"])
parser.add_argument('--l', default = 0.05, type = float)


args = parser.parse_args()



def main():
    input_data = pickle.load(open("./dataset/{}/train.pkl".format(args.dataset),"rb"))
    target_data = pickle.load(open("./dataset/{}/test.pkl".format(args.dataset),"rb"))

    dict_theta = {'theta_0_p':args.c,'theta_0_m':0,'theta_1_p':0,'theta_1_m':args.c}
    print("Corrupt ratio: {}".format(args.c))

    training_generator,validation_generator,testing_generator=load_data_generator(input_data,target_data,dict_theta,args.batch_size,args.seed,c_num=args.c_num)

    x_dim = input_data['x'].shape
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")


    #==================== Validation model #====================
  
    if args.retrain_validation == "retrain":
        print("start zero shot predictor estimation.")
        irr_loss = zero_shot_predictor_loss(training_generator,device,args,alpha=args.alpha,gamma=args.gamma)
        torch.save(irr_loss, "irred_losses_and_checks.pt")
    else:
        print("skip and load.")
        pass
     
    irr_loss_generator = torch.load("irred_losses_and_checks.pt")
    train_ratio = args.train_ratio
    print("save irr loss generator")
    #=============================================================

    loss_function = nn.CrossEntropyLoss(reduction="none")
    flip_flag = False
    
    if args.dataset == "lfwa_w":
        resnet = models.resnet18(pretrained=False)
    else:
        resnet = models.resnet50(pretrained=False)
  
    resnet.fc = nn.Linear(resnet.fc.in_features,args.num_class)
    resnet.to(device)

    opt_model = torch.optim.Adam(resnet.parameters(), lr=args.lr,
                                 weight_decay=0.01, 
                                 amsgrad=True, eps=args.opt_eps)
    print("START")
    for t in range(0,args.num_iter):
        for i, batch in enumerate(training_generator):
            clean_num, train_num = training_step(resnet,batch,opt_model,train_ratio,irr_loss_generator,loss_function,args,flip=flip_flag)
    resnet.eval()
    results = test_step(resnet, testing_generator,flip=flip_flag)
    df = pd.DataFrame(results,index=[0]).round(3)
    print(df)

    return results


if __name__ == "__main__":

    acc_list = []
    dp_list = []
    deo_list = []
    f1_list = []
    di_list = []

    for i in range(3):
        res = main()
        acc_list.append(res['acc'])
        dp_list.append(res['dp'])
        deo_list.append(res['deo'])
        f1_list.append(res['f1'])
        di_list.append(res['di'])

    acc_ = np.array(acc_list)
    dp_ = np.array(dp_list)
    f1_ = np.array(f1_list)
    deo_ = np.array(deo_list)
    di_ = np.array(di_list)

    print("acc : {:.2f} +/- {:.2f}".format(np.mean(acc_),np.std(acc_)))
    print("dp : {:.2f} +/- {:.2f}".format(np.mean(dp_),np.std(dp_)))
    print("deo : {:.2f} +/- {:.2f}".format(np.mean(deo_),np.std(deo_)))
    print("di : {:.2f} +/- {:.2f}".format(np.mean(di_)*100,np.std(di_)*100))
    print("f1 : {:.2f} +/- {:.2f}".format(np.mean(f1_),np.std(f1_)))
