import torch
import numpy as np
import cifar_input as cifar_data
import cat_and_dog
import stl10 as stl10_data
import torchvision.models
from sklearn import metrics
import argparse
import time

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--loss',  type=str, default='ce',
                    help='an integer for the accumulator')
parser.add_argument('--lr0',  type=float, default=0.1,
                    help='an integer for the accumulator')
parser.add_argument('--keep_index',  type=float, default=1.0,
                    help='an integer for the accumulator')
parser.add_argument('--eval_interval',  type=int, default=400,
                    help='an integer for the accumulator')
parser.add_argument('--dataset',  type=int, default=2,
                    help='an integer for the accumulator')
parser.add_argument('--split_index',  type=int, default=4,
                    help='an integer for the accumulator')
parser.add_argument('--c',  type=int, default=2000,
                    help='an integer for the accumulator')
parser.add_argument('--is_stl10',  type=bool, default=False,
                    help='an integer for the accumulator')

args = parser.parse_args()

def AUC(y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)

def heaviside(x):
    return 1 / (1 + torch.exp(-100*x))

keep_index = args.keep_index
is_tune = False
is_crop_flip = False
batch_size = 32
total_num_epochs = 20
bsize = 1 # batch size for  LP solver. 1 since one batch merge into 1 set of positive / negative samples
epsilon = 0.01 # epsilon for AUC
step_size = 0.1
lr0 = args.lr0

# Import CIFAR data
if args.dataset != 2 and args.is_stl10 == False:
    if args.dataset == 10:
        chosen_dataset = 'cifar10'
    else:
        chosen_dataset = 'cifar100'
    (train_data, train_labels), (test_data, test_labels) = cifar_data.load_data(args.dataset, False,
                                                                                False)
    split_index = args.split_index if args.dataset == 10 else args.split_index
    train_labels[train_labels <= split_index] = -1  # [0, ....]
    test_labels[test_labels <= split_index] = -1
    train_labels[train_labels >= split_index + 1] = 1  # [0, ....]
    test_labels[test_labels >= split_index + 1] = 1

    train_ids = list(range(train_data.shape[0]))
    np.random.seed(123)
    np.random.shuffle(train_ids)
    train_data = train_data[train_ids]
    train_labels = train_labels[train_ids]

    # delete some samples
    num_neg = np.where(train_labels == -1)[0].shape[0]
    idx_neg_tmp = np.where(train_labels == -1)[0][:int(num_neg * args.keep_index)]
    idx_pos_tmp = np.where(train_labels == 1)[0]
    train_data = train_data[idx_neg_tmp.tolist() + idx_pos_tmp.tolist()]
    train_labels = train_labels[idx_neg_tmp.tolist() + idx_pos_tmp.tolist()]

    pos_count = np.count_nonzero(train_labels == 1)
    neg_count = np.count_nonzero(train_labels == -1)
    print('Pos:Neg: [%d : %d]' % (np.count_nonzero(train_labels == 1), np.count_nonzero(train_labels == -1)))

if args.dataset == 2 and args.is_stl10 == False:
    chosen_dataset = 'cat_and_dog'
    (train_data, train_labels), (test_data, test_labels) = cat_and_dog.load_data(args.dataset,False,
                                                                                False)

    train_ids = list(range(train_data.shape[0]))
    np.random.seed(123)
    np.random.shuffle(train_ids)
    train_data = train_data[train_ids]
    train_labels = train_labels[train_ids]
    split_index = 0

    train_labels[train_labels <= split_index] = -1  # [0, ....]
    test_labels[test_labels <= split_index] = -1
    train_labels[train_labels >= split_index + 1] = 1  # [0, ....]
    test_labels[test_labels >= split_index + 1] = 1

    # delete some samples
    num_neg = np.where(train_labels == -1)[0].shape[0]
    idx_neg_tmp = np.where(train_labels == -1)[0][:int(num_neg * args.keep_index)]
    idx_pos_tmp = np.where(train_labels == 1)[0]
    train_data = train_data[idx_neg_tmp.tolist() + idx_pos_tmp.tolist()]
    train_labels = train_labels[idx_neg_tmp.tolist() + idx_pos_tmp.tolist()]

    train_labels[train_labels == 0] = -1
    test_labels[test_labels == 0] = -1
    pos_count = np.count_nonzero(train_labels == 1)
    neg_count = np.count_nonzero(train_labels == -1)
    print('Pos:Neg: [%d : %d]' % (np.count_nonzero(train_labels == 1), np.count_nonzero(train_labels == -1)))

# Import CIFAR10 +
if args.dataset != 2 and args.is_stl10 != False:
    chosen_dataset = 'STL'
    (train_data, train_labels), (test_data, test_labels) = stl10_data.load_data(args.dataset, False,
                                                                                False)
    split_index = args.split_index if args.dataset == 10 else args.split_index
    train_labels[train_labels <= split_index] = -1  # [0, ....]
    test_labels[test_labels <= split_index] = -1
    train_labels[train_labels >= split_index + 1] = 1  # [0, ....]
    test_labels[test_labels >= split_index + 1] = 1

    train_ids = list(range(train_data.shape[0]))
    np.random.seed(123)
    np.random.shuffle(train_ids)
    train_data = train_data[train_ids]
    train_labels = train_labels[train_ids]

    # delete some samples
    num_neg = np.where(train_labels == -1)[0].shape[0]
    idx_neg_tmp = np.where(train_labels == -1)[0][:int(num_neg * args.keep_index)]
    idx_pos_tmp = np.where(train_labels == 1)[0]
    train_data = train_data[idx_neg_tmp.tolist() + idx_pos_tmp.tolist()]
    train_labels = train_labels[idx_neg_tmp.tolist() + idx_pos_tmp.tolist()]

    pos_count = np.count_nonzero(train_labels == 1)
    neg_count = np.count_nonzero(train_labels == -1)
    print('Pos:Neg: [%d : %d]' % (np.count_nonzero(train_labels == 1), np.count_nonzero(train_labels == -1)))

# shuffle data
train_ids = list(range(train_data.shape[0]))
np.random.seed(None)
np.random.shuffle(train_ids)
train_data = train_data[train_ids]
train_labels = train_labels[train_ids]
test_auc = []
test_iter = []
total_iter = 0
num_batch = train_labels.shape[0]//batch_size

relu = torch.nn.ReLU()

# get resnet model
resnet18 = torchvision.models.resnet18(pretrained=False)
resnet18.fc = torch.nn.Linear(512, 2)
resnet18 = resnet18.cuda()
softmax = torch.nn.Softmax(dim=1)
cross_entropy = torch.nn.CrossEntropyLoss()

# optimize
optimizer = torch.optim.SGD(resnet18.parameters(), lr=lr0, momentum=0.9)#0.0005

test_auc_list = []
break_flag = False
time_list = []
for k in range(1, 1000):
    if break_flag:
        break

    T_k = args.c * (3 ** (k - 1))
    lr_k = lr0 * (1 / 3 ** (k - 1))
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr_k
    for t in range(1, int(T_k)+1):
        total_iter += 1
        t_s = time.time()

        # calculate AUC on test set
        if total_iter % args.eval_interval == 0 or total_iter==1:
            images, labels = test_data, test_labels
            test_num_batches = images.shape[0] // batch_size
            pred_probs = []
            for step in range(test_num_batches):
                offset = step * batch_size
                vali_data_batch = images[offset:offset + batch_size]
                vali_label_batch = labels[offset:offset + batch_size][:, np.newaxis]
                batch_x = torch.from_numpy(vali_data_batch).float().cuda().permute(0, 3, 1, 2)
                ### predict logits using resnet18
                logits = resnet18(batch_x)  # channel before grid size
                logits = softmax(logits)[:,1] #torch.sigmoid(logits.squeeze())  # transform into range 0-1
                pred_probs.extend(logits.tolist())

            auc = AUC(test_labels[:test_num_batches * batch_size], pred_probs)
            print('Test AUC:', auc, '# of iters:', total_iter)
            test_auc_list.append(auc)

        idx = total_iter % num_batch
        if idx == 0:  # shuffle dataset every epoch
            np.random.shuffle(train_ids)
            train_data = train_data[train_ids]
            train_labels = train_labels[train_ids]
            # continue
            idx += 1

        offset = (idx - 1) * batch_size
        batch_x, batch_y = (train_data[offset:offset + batch_size], train_labels[offset:offset + batch_size][:, np.newaxis])
        batch_x = torch.from_numpy(batch_x).float().cuda().permute(0,3,1,2)
        batch_y = torch.from_numpy(batch_y).long().cuda().squeeze()
        ### predict logits using resnet18
        logits_raw = resnet18(batch_x) # channel before grid size
        logits = softmax(logits_raw)[:,1] #torch.sigmoid(logits.squeeze()) # transform into range 0-1

        pos_idxs = np.nonzero((batch_y == 1).float().cpu().numpy())[0]
        neg_idxs = np.nonzero((batch_y == -1).float().cpu().numpy())[0]
        scores_pos = logits[pos_idxs].unsqueeze(0)
        scores_neg = logits[neg_idxs].unsqueeze(0)
        T = scores_pos.shape[1]
        N = scores_neg.shape[1]

        scores_pos_minus_neg = scores_pos.unsqueeze(2) - scores_neg.unsqueeze(1) # shape=[batch_size, T, N]

        if args.loss == 'auc':
            ### solve using Olvi's method
            c = torch.ones([bsize, T * N]).cuda()
            A = torch.cat([-torch.eye(T * N), -torch.eye(T * N)], 0).unsqueeze(0).repeat(bsize, 1, 1).cuda()  # shape=[batch_size, T*N, 2*T*N]
            b =  torch.cat([scores_pos_minus_neg.view(scores_pos_minus_neg.shape[0], -1).cuda() - epsilon,
                            torch.zeros(bsize, T*N).cuda()], 1)
            #initialize x0
            n = A.shape[2]
            xs,_ = torch.solve(A[:,:n,:].transpose(1,2).bmm(b[:,:n].unsqueeze(2)),
                               A[:,:n,:].transpose(1,2).bmm(A[:,:n,:])+0.001*torch.eye(n).unsqueeze(0).repeat(A.shape[0],1,1).cuda())
            for i in range(20):
                tmp = (torch.diag_embed(A.bmm(xs).squeeze(2) - b) > 0).float() #hard version of step function
                #tmp = heaviside(torch.diag_embed(A.bmm(xs).squeeze(2) - b)) #soft version of step function
                P = A.transpose(1,2).bmm(tmp).bmm(A) + 0.0001*torch.eye(A.shape[2]).cuda().unsqueeze(0).repeat(A.shape[0],1,1)
                Q = A.transpose(1,2).bmm(relu(A.bmm(xs) - b.unsqueeze(2))) + 0.001*c.unsqueeze(2)
                d = - torch.inverse(P).bmm(Q)
                xs = xs + d # original step size is 1. Try 0.1, 0.2, 0.5 for ablation
            v = 1/epsilon * relu(A.bmm(xs) -b.unsqueeze(2))
            v_gr_0 = (v>0).float().squeeze().nonzero().squeeze(1)
            A_j = A[:,v_gr_0, :]
            b_j = b[:,v_gr_0]
            xs, _ = torch.solve(A_j.transpose(1,2).bmm(b_j.unsqueeze(2)),
                             A_j.transpose(1,2).bmm(A_j)+0.01*torch.eye(A_j.shape[2]).unsqueeze(0).repeat(A_j.shape[0],1,1).cuda())
            z_sol = xs.squeeze(2)
            print((z_sol < epsilon).float().sum(), (scores_pos_minus_neg > 0).float().sum(), T * N)
        #################################

        # calculate AUC loss
        if args.loss == 'auc':
            n = T * N
            loss = (n - relu(-z_sol + epsilon).sum(1) / epsilon ) / T / N
            loss = loss.mean() # mean for beach

        # cross entropy loss
        if args.loss == 'ce':
            loss = cross_entropy(logits_raw, (batch_y==1).long())

        # return loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if total_iter > 50:
            time_list.append(time.time() - t_s)
            print('Iteration:', total_iter, 'loss:', loss.item(), 'time:', np.mean(np.array(time_list)))
        if total_iter >= 40001:
            break_flag = True
            break

print('finish')
