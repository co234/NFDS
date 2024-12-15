import numpy as np
from torch.utils import data
import torch
import torch.nn.functional as F
import pickle
import math
from sklearn.utils import resample
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")

def fairness_measure(label,pred_label,sensitive_feature,type="di"):
    pred_label = np.array(pred_label)
    sensitive_feature = np.array(sensitive_feature)
    label= np.array(label)

    a_idx,b_idx = np.where(sensitive_feature==0)[0], np.where(sensitive_feature==1)[0]
    a_ratio, b_ratio = (pred_label[a_idx] == label[a_idx]).sum()/len(a_idx), \
                        (pred_label[b_idx] == label[b_idx]).sum()/len(b_idx)
    

    if type == "dp":
        f_score = np.abs(a_ratio-b_ratio)

    elif type == "eo":
        y_pos = np.where(label==1)[0]
        pos_a,pos_b = np.intersect1d(y_pos,a_idx), np.intersect1d(y_pos,b_idx)
        a_eo,b_eo = (pred_label[pos_a]==1).sum()/(len(pos_a)+1e-15), (pred_label[pos_b]==1).sum()/(len(pos_b)+1e-15)
        f_score = np.abs(a_eo-b_eo)

    elif type == "di":
        f_score = np.min([(a_ratio/(b_ratio+1e-15)),(b_ratio/(a_ratio+1e-15))])

    else:
        print("NotImplement")
        f_score = None
    
    return f_score

    

def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)

def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld

def split_data(x,y,yt,a,test_size):
    np.random.seed(129)

    n = x.shape[0]
    idx = list(np.random.permutation(n))
    train_size = int(n*(1-test_size))
    test_size = int(n*test_size)

    shuffled_X,shuffled_y,shuffled_a, shuffled_yt = x[idx,:],y[idx],a[idx],yt[idx]

    X_train,y_train,a_train,yt_train =  shuffled_X[:train_size,:],\
                                        shuffled_y[:train_size],\
                                        shuffled_a[:train_size],\
                                        shuffled_yt[:train_size]
    
    X_test,y_test,a_test,yt_test= shuffled_X[train_size:,:],\
                                  shuffled_y[train_size:],\
                                  shuffled_a[train_size:],\
                                  shuffled_yt[train_size:]

    print("clean train set di: ", fairness_measure(y_train,y_train,a_train))
    print("clean test set di: ", fairness_measure(y_test,y_test,a_test))
    print("noisy train set di: ",fairness_measure(yt_train,yt_train,a_train))
    print("noisy test set di: ", fairness_measure(yt_test,yt_test,a_test))
    
    
    input_data = {'x': X_train, 'y': y_train, 'a': a_train, 'yt':yt_train}
    target_data = {'x': X_test, 'y': y_test, 'a': a_test, 'yt':yt_test}

    return input_data, target_data


class Dataset():
    def __init__(self, x, labels, sensitive_attribute,clean_label):
        self.x = x
        self.labels = labels
        self.sensitive_attribute = sensitive_attribute
        self.clean_label = clean_label
        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return int(index), self.x[index], int(self.labels[index]), int(self.sensitive_attribute[index]), int(self.clean_label[index])
    


def add_label_bias(yclean,rho,theta_dict,seed):
    """
    theta_0_p: P(Y=+1|Z=-1,A=0)
    theta_0_m: P(Y=-1|Z=+1,A=0)
    theta_1_p: P(Y=+1|Z=-1,A=1)
    theta_1_m: P(Y=-1|Z=+1,A=1)
    """
    n = len(yclean)
    np.random.seed(seed)

    t_0_p, t_0_m, t_1_p,t_1_m = theta_dict['theta_0_p'],theta_dict['theta_0_m'],theta_dict['theta_1_p'],theta_dict['theta_1_m']


    def locate_group(label,sensitive_attr,a,y):
        return np.intersect1d(np.where(sensitive_attr==a)[0],np.where(label==y)[0])

    g_01, g_00 = locate_group(yclean,rho,0,1),locate_group(yclean,rho,0,0)
    g_11, g_10 = locate_group(yclean,rho,1,1),locate_group(yclean,rho,1,0)

    group = [g_01,g_00,g_11,g_10]
    theta = [t_0_m,t_0_p,t_1_m,t_1_p]
    tilde_y = [0,1,0,1]

    t = yclean.copy()

    for i in range(len(group)):
        for j in range(len(group[i])):
            p = np.random.uniform(0,1)
            if p < theta[i]:
                t[group[i][j]] = tilde_y[i]
            else:
                t[group[i][j]] = yclean[group[i][j]]



    return t



def add_individual_bias(x,y,c_ratio = 0.1):
    svc = svm.SVC(kernel="linear",max_iter=150).fit(x,y)
    score = svc.decision_function(x)
    w_norm = np.linalg.norm(svc.coef_)
    dist = score/w_norm

    p = 1/np.abs(dist)
    idx = np.random.choice(len(x),int(c_ratio*len(x)),list(p))

    y_noise = []
    for i in range(len(y)):
        if i in idx:
            if y[i] ==0:
                y_=1
            elif y[i] == 1:
                y_=0
        else:
            y_ = y[i]

        y_noise.append(y_)

    return y_noise


def flip_inctance(x,y,c_ratio = 0.1):
    p = 1/len(x)
    idx = np.random.choice(len(x),int(c_ratio*len(x)))
    
    y_noise = []
    for i in range(len(y)):
        if i in idx:
            if y[i] ==0:
                y_=1
            elif y[i] == 1:
                y_=0
        else:
            y_ = y[i]

        y_noise.append(y_)

    return y_noise


def load_data_generator(input_data,target_data,noise_dict,batch_size=64,seed=64,c_num=100):
    np.random.seed(42)


    y = input_data['y'].squeeze().long().cpu().detach().numpy()
    s = input_data['s'].cpu().detach().numpy()

    y_new = add_label_bias(y,s,noise_dict,seed)
    y_tilder = torch.as_tensor(y_new, dtype=torch.float32).unsqueeze(-1)
    input_data['yt'] = y_tilder

    idx_to_meta, idx_to_train = prepare_data(input_data,c_num)
    
    train_silver = Dataset(input_data['x'][idx_to_train], \
                           input_data['yt'][idx_to_train], \
                           input_data['s'][idx_to_train], 
                           input_data['y'][idx_to_train])
    train_loader = data.DataLoader(train_silver, batch_size=batch_size,shuffle=True)
    
    train_gold = Dataset(input_data['x'][idx_to_meta], \
                           input_data['y'][idx_to_meta], \
                           input_data['s'][idx_to_meta],\
                            input_data['y'][idx_to_meta])
    validation_loader = data.DataLoader(train_gold, batch_size=batch_size,shuffle=True)


    testing_set = Dataset(target_data['x'], target_data['y'], target_data['s'],target_data['y'])
    test_loader = data.DataLoader(testing_set, batch_size=batch_size,shuffle=True)

    return train_loader,validation_loader, test_loader





class DataIterator(object):
    def __init__(self, dataloader):
        assert isinstance(dataloader, torch.utils.data.DataLoader), 'Wrong loader type'
        self.loader = dataloader
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            x, y,a = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            x,y,a = next(self.iterator)

        return x,y,a

def _compute_expectation(pred,Y):
    exp_loss = torch.zeros(pred.shape[0],)
    for i in range(len(pred)):
        if torch.all(pred[i] == 0):
            exp_loss[i] = 0
        else:
            temp_matrix = pred[i].repeat(Y.shape[0], 1)
            Y_long = Y.long()
            exp_loss[i] = F.cross_entropy(temp_matrix, Y_long, reduction="mean") 

    return exp_loss


def _fair_peer_prediction(sensitive_attr,y_pred,y_true):
    m_idx, f_idx = np.where(sensitive_attr.cpu()==0)[0], np.where(sensitive_attr.cpu()==1)[0]
    y_m = torch.tensor(y_true[m_idx],dtype=torch.float)
    y_f = torch.tensor(y_true[f_idx],dtype=torch.float)
    sens = sensitive_attr.unsqueeze(1)
    Ea = _compute_expectation((1-sens)*(y_pred),y_m)
    Eb = _compute_expectation(sens*(y_pred),y_f)

    return Ea+Eb



def load_data(input_data,target_data,noise_dict,batch_size=64,seed=1234,c_num=1000):

    # np.random.seed(1234)

    y = input_data['y'].squeeze().long().cpu().detach().numpy()
    s = input_data['s'].cpu().detach().numpy()

    y_new = add_label_bias(y,s,noise_dict,seed)
    y_tilder = torch.as_tensor(y_new, dtype=torch.float32).unsqueeze(-1)
    input_data['yt'] = y_tilder

    idx_to_meta, idx_to_train = prepare_data(input_data,c_num)
    idx_to_test = [i for i in range(len(target_data['x']))]

    
    train_data = Dataset(input_data['x'][idx_to_train], \
                           input_data['yt'][idx_to_train], \
                           input_data['s'][idx_to_train])
    train_loader = data.DataLoader(idx_to_train, batch_size=batch_size,shuffle=True)
    
    valid_data = Dataset(input_data['x'][idx_to_meta], \
                           input_data['y'][idx_to_meta], \
                           input_data['s'][idx_to_meta])
    valid_loader = data.DataLoader(idx_to_meta,batch_size=batch_size,shuffle=True )


    test_data = Dataset(target_data['x'], target_data['y'], target_data['s'])
    test_loader = data.DataLoader(idx_to_test,batch_size=batch_size,shuffle=True )



    return train_loader, valid_loader, test_loader



def prepare_data(input_data,c_ratio=0.01):
    # np.random.seed(42)
    idx_to_meta = []
    idx_to_train = []

    num = int(len(input_data['y'])*c_ratio/4)

    data_label_val = {}
    for j in range(2):
        data_label_val[j] = [i for i, label in enumerate(input_data['y']) if label == j]

    data_sens_val = {}
    for j in range(2):
        data_sens_val[j] = [i for i, sens in enumerate(input_data['s']) if sens == j]


    for cls_label, cls_list in data_label_val.items():
        for sens_label, sens_list in data_sens_val.items():
            tmp_list = np.intersect1d(cls_list,sens_list)
            np.random.shuffle(tmp_list)
            idx_to_meta.extend(tmp_list[:num])
            idx_to_train.extend(tmp_list[num:])

    print("num of meta data: {}, num of train data: {}".format(len(idx_to_meta),len(idx_to_train)))

    return idx_to_meta, idx_to_train



def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    """
    Compute conditional entropy
    Args:
        log_probs_N_K_C: torch.Tensor, the log probabilities. With the following ordering:
            N: Batch Size
            K: Monte-Carlo Samples
            C: Number of Classes
    Returns:
        entropies_N: torch.Tensor, conditional entropy for each sample in batch
    """
    N, K, C = log_probs_N_K_C.shape

    nats_N_K_C = log_probs_N_K_C * torch.exp(log_probs_N_K_C)
    entropies_N = -torch.sum(nats_N_K_C, dim=(1, 2)) / K # simply average across MC samples

    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy
    Args:
        log_probs_N_K_C: torch.Tensor, the log probabilities. With the following ordering:
            N: Batch Size
            K: Monte-Carlo Samples
            C: Number of Classes
    Returns:
        entropies_N: torch.Tensor, entropy for each sample in batch
    """
    # N (batch size), K (MC), C (Classes)
    N, K, C = log_probs_N_K_C.shape

    mean_log_probs_N_C = torch.logsumexp(log_probs_N_K_C, dim=1) - math.log(K) # average over posterior samples
    nats_N_C = mean_log_probs_N_C * torch.exp(mean_log_probs_N_C)
    entropies_N = -torch.sum(nats_N_C, dim=1)

    return entropies_N


def get_bald(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    """
    Return BALD score
    Args:
        log_probs_N_K_C: torch.Tensor, the log probabilities. With the following ordering:
            N: Batch Size
            K: Monte-Carlo Samples
            C: Number of Classes
    Returns:
        scores_N: torch.Tensor, bald scores for each sample in batch

        Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Máté Lengyel. Bayesian active learning for classification and preference learning. arXiv preprint arXiv:1112.5745, 2011.
    """
    # N (batch size), K (MC), C (Classes)
    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)

    return scores_N


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()



import torch


def top_x_indices(vec, x, largest):
    """
    Returns the indices of the x largest/smallest entries in vec.

    Args:
        vec: tensor, number of samples to be selected
        x: int, number of indices to be returned
        smallest: bool, if true, the x largest entries are selected; if false,
        the x smallest entries are selected
    Returns:
        top_x_indices: tensor, top x indices, sorted
        other_indices: tensor, the indices that were not selected
    """

    sorted_idx = torch.argsort(vec, descending=largest)

    top_x_indices = sorted_idx[:x]
    other_indices = sorted_idx[x:]

    return top_x_indices, other_indices





def create_logging_dict(variables_to_log, selected_minibatch, not_selected_minibatch):
    """
    Create the dictionary for logging, in which, for each variable/metric, the
    selected and the not selected entries are logged separately as
    "selected_<var_name>" and "not_selected_<var_name>".

    Args:
        variables_to-log: dict, with key:var_name to be logged, value: tensor of values to be logger.
    """

    metrics_to_log = {}
    for name, metric in variables_to_log.items():
        metrics_to_log["selected_" + name] = metric[selected_minibatch].cpu().numpy()
        metrics_to_log["not_selected_" + name] = (
            metric[not_selected_minibatch].cpu().numpy()
        )
    return metrics_to_log


def compute_selection_bias(data, label, sens_attribute):
    n_a, n_b = len(np.where(sens_attribute==0)[0]), len(np.where(sens_attribute==1)[0])
    n_p, n_n = len(np.where(label==1)[0]), len(np.where(label==0)[0])

    n = len(label)

    n_a_p = len(np.intersect1d(np.where(sens_attribute==0)[0], np.where(label==1)[0]))
    n_a_n = len(np.intersect1d(np.where(sens_attribute==0)[0], np.where(label==0)[0]))

    n_b_p = len(np.intersect1d(np.where(sens_attribute==1)[0], np.where(label==1)[0]))
    n_b_n = len(np.intersect1d(np.where(sens_attribute==1)[0], np.where(label==0)[0]))


    diff = n_a_p - n_a/(n+1e-6) * n_p
    return diff


def sampling(data, label, sens_attribute, clean_label, diff):
    n = int(np.abs(diff))
    if diff < 0:
        idx = np.random.choice(len(label),n,replace=True)
        data_ = torch.cat((data,data[idx]),0)
        label_ = torch.cat((label,label[idx]),0)
        clean_label_ = torch.cat((clean_label,clean_label[idx]),0)
        sens_attribute_ = torch.cat((sens_attribute,sens_attribute[idx]),0)
        
        
    else:
        idx = np.random.choice(len(label),len(data)-n,replace=False)
        data_ = data[idx]
        label_ = label[idx]
        sens_attribute_ = sens_attribute[idx]
        clean_label_ = clean_label[idx]

    

    return data_,label_,sens_attribute_,clean_label_


 
