from ast import parse
import numpy as np
import os
import argparse, sys
import datetime
import shutil
import copy
from numpy.lib.utils import source
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from reducible_loss import reducible_loss_selection
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import time
import pickle
from utils import _fair_peer_prediction, compute_selection_bias,sampling,fairness_measure
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
import clip





def training(model, train_loader, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=0, 
                                 amsgrad=True, eps=args.opt_eps)


    train_total = 0
    train_correct = 0

    for d in train_loader:

        data = d[1].to(device)
        labels = d[2].to(device)
        
        optimizer.zero_grad()
        logits = model(data)
        out =  torch.argmax(logits, dim=1)
        acc = accuracy_score(out.cpu(),labels.cpu())
        train_total +=1
        train_correct+=acc 

        loss = F.cross_entropy(logits,labels)
        loss.backward()
        optimizer.step()
    
    train_acc = float(train_correct)/float(train_total)

    return train_acc


def training_step(model,batch,opt_model,tarin_ratio,irreducible_loss_generator,loss_function,args,flip=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_idx, data, target, sensitive_proxy, clean_label = batch[0], batch[1].to(device), batch[2].to(device), batch[3], batch[4].to(device)
    batch_size = len(data)

    selected_batch_size = max(1, int(batch_size*tarin_ratio))


    selection_function = reducible_loss_selection()


    selected_indices = selection_function.__call__(
            selected_batch_size=selected_batch_size,
            data=data,
            target=target,
            sensitive_attrs = sensitive_proxy,
            global_index=global_idx,
            large_model=model,
            irreducible_loss_generator=irreducible_loss_generator,
            l = args.l
        )  

            

    model.train()  
    data_selected, target_selected, sens_selected, clean_selected = data[selected_indices], \
                                    target[selected_indices], \
                                    sensitive_proxy[selected_indices.cpu()],\
                                    clean_label[selected_indices]

    
    if args.selection == "sampling":
        diff = compute_selection_bias(data_selected.cpu(),target_selected.cpu(),sens_selected.cpu())
        data_final, target_final, sens_final, clean_label_final = sampling(data_selected,target_selected,sens_selected,clean_selected,diff)
        clean_idx = np.where(target_final.cpu() == clean_label_final.cpu())[0]
    else:
        data_final, target_final, sens_final, clean_label_final = data_selected, target_selected, sens_selected, clean_selected
        clean_idx = np.where(target_final.cpu() == clean_label_final.cpu())[0]


    opt_model.zero_grad()
    
    logits = model(data_final.to(device))
    loss = loss_function(logits,target_final.to(device))
    loss.mean().backward()
    opt_model.step()

    

    return len(clean_idx), len(target_final)




def validation_step(model,validation_set, flip = False):
    correct = 0
    total = 0

    for batch in validation_set:
        data, target = batch[1], batch[2]
        logits = model(data)
        # training metrics
        if flip is True:
            preds = logits.data>=0
            preds = preds.view(-1).long()
        else:
            preds = torch.argmax(F.log_softmax(logits,dim=1),dim=1)
        
        total += target.size(0)
        correct += (preds == target.long()).sum()


    acc = 100*float(correct)/float(total)

    return acc



def test_step(model,test_set,flip=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prediction = []
    ground_truth = []
    sensitive_attrs = []
    for batch in test_set:
        data, target, sens_proxy = batch[1].to(device), batch[2].to(device), batch[3]
        logits = model(data)
        
        if flip is True:
            preds = logits.data>=0
            preds = preds.view(-1).long()
        else:
            preds = torch.argmax(F.log_softmax(logits,dim=1),dim=1)


        prediction+=list(preds.cpu().detach().numpy())
        ground_truth+=list(target.cpu().detach().numpy())
        sensitive_attrs+=list(sens_proxy.detach().numpy())

    f1 = 100*f1_score(ground_truth,prediction,average='weighted')
    acc = 100*accuracy_score(ground_truth,prediction)

    res = {'dp':fairness_measure(ground_truth,prediction,sensitive_attrs,type="dp"),
           'deo':fairness_measure(ground_truth,prediction,sensitive_attrs,type="eo"),
           'di': fairness_measure(ground_truth,prediction,sensitive_attrs,type="di"),
           'acc':acc,
           'f1':f1
    }


    return res
    

def testing(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    for d in test_loader:
        data, label = d[1].to(device), d[2].to(device)
        pred = model(data)
        out = torch.argmax(F.log_softmax(pred,dim=1),dim=1)

        total += label.size(0)
        correct += (out == label.long()).sum()   
    acc = 100*float(correct)/float(total)
    
    return acc




def zero_shot_predictor_loss(dataloader, device, args, alpha=0.1, gamma=0.9):
    losses = []
    idxs = []
    targets = []
    prediction_correct = []
    sens_idxs = []
    
    model, preprocess = clip.load("RN50", device=device)

    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        preprocess.transforms[1]  
    ])
    

    for d in dataloader:
        idx, x, target,sens = d[0], d[1], d[2].to(device), d[3].to(device)
        resized_img = resize_transform(x).to(device)
        if args.dataset == "celeba":
            class_names = ["no heavy makeup","heavy makeup"]
        else:
            class_names = ["female","male"]
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {name}") for name in class_names]).to(device)
        
        with torch.no_grad():
            img_features = model.encode_image(resized_img)
            text_features = model.encode_text(text_inputs)

            img_features /= img_features.norm(dim=-1,keepdim=True)
            text_features /= text_features.norm(dim=-1,keepdim=True)

            similarity = img_features @ text_features.T
            loss = F.cross_entropy(similarity, target, reduction="none")
            
            expected_part = _fair_peer_prediction(sens,similarity,target).to(device)
            new_loss = (1-torch.tensor(alpha).to(device))*loss - torch.tensor(gamma).to(device)*expected_part 
            losses.append(new_loss)
            idxs.append(idx)
            sens_idxs.append(sens)
            targets.append(target)
            prediction_correct.append(torch.eq(torch.argmax(similarity, dim=1), target))
        
    acc = torch.cat(prediction_correct, dim=0)
    acc = acc.type(torch.FloatTensor).mean()
    print("clip acc: {:.2f}".format(acc))
    average_loss = torch.cat(losses, dim=0).type(torch.FloatTensor).mean()
    losses_temp = torch.cat(losses, dim=0).to(device)
    idxs = torch.cat(idxs, dim=0)
    targets_temp = torch.cat(targets, dim=0)

    max_idx = idxs.max()

    losses = torch.tensor(
        [float("nan")] * (max_idx + 1), dtype=losses_temp.dtype
    )  
    targets = torch.zeros(max_idx + 1, dtype=targets_temp.dtype)
    losses.cpu()[idxs] = losses_temp.cpu()
    targets.cpu()[idxs] = targets_temp.cpu()

    output = {
        "irreducible_losses": losses.to(device),
        "sorted_targets": targets,
        "heldout_accuracy": acc,
        "heldout_average_loss": average_loss,
    }

    return output


def compute_losses_with_sanity_checks(dataloader, model, device):
    losses = []
    idxs = []
    targets = []
    prediction_correct = []

    with torch.no_grad():
        for d in dataloader:
            idx, x, target = d[0], d[1].to(device), d[2].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, target, reduction="none")
            losses.append(loss)
            idxs.append(idx)
            targets.append(target)
            prediction_correct.append(torch.eq(torch.argmax(logits, dim=1), target))
    
        
    acc = torch.cat(prediction_correct, dim=0)
    
    acc = acc.type(torch.FloatTensor).mean()
    
    average_loss = torch.cat(losses, dim=0).type(torch.FloatTensor).mean()

    losses_temp = torch.cat(losses, dim=0)
    idxs = torch.cat(idxs, dim=0)
    targets_temp = torch.cat(targets, dim=0)

    max_idx = idxs.max()

    losses = torch.tensor(
        [float("nan")] * (max_idx + 1), dtype=losses_temp.dtype
    )  # losses[global_index] is the irreducible loss of the datapoint with that global index. losses[idx] is nan if idx is not part of the dataset.
    targets = torch.zeros(max_idx + 1, dtype=targets_temp.dtype)
    losses.cpu()[idxs] = losses_temp.cpu()
    targets.cpu()[idxs] = targets_temp.cpu()

    output = {
        "irreducible_losses": losses.to(device),
        "sorted_targets": targets,
        "heldout_accuracy": acc,
        "heldout_average_loss": average_loss,
    }

    return output



def evaluate_and_save_model(model,path_name,training_generator,device):
    model.load_state_dict(torch.load(path_name))
    model.eval()
    irr_loss = compute_losses_with_sanity_checks(training_generator,model,device)

    torch.save(irr_loss, "irred_losses_and_checks.pt")
