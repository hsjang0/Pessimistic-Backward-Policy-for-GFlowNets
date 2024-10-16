import torch as T
import numpy as np
import tqdm
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib
import wandb
import io
from PIL import Image

import argparse

device = T.device('cuda')

horizon = 16
ndim = 3

n_hid = 256
n_layers = 2

bs = 64

save_name='our'
wandb.init(project='grid', name='PBP-GFN')

def make_mlp(l, act=T.nn.LeakyReLU(), tail=[]):
    return T.nn.Sequential(*(sum(
        [[T.nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

def log_reward(x):
    ax = abs(x / (horizon-1) * 2 - 1)
    return ((ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-3).log()

j = T.zeros((horizon,)*ndim+(ndim,))
for i in range(ndim):
    jj = T.linspace(0,horizon-1,horizon)
    for _ in range(i): jj = jj.unsqueeze(1)
    j[...,i] = jj
truelr = log_reward(j)
print('total reward', truelr.view(-1).logsumexp(0))
true_dist = truelr.flatten().softmax(0).cpu().numpy()
def toin(z):
    return T.nn.functional.one_hot(z,horizon).view(z.shape[0],-1).float()
Z = T.zeros((1,)).to(device)
model = make_mlp([ndim*horizon] + [n_hid] * n_layers + [ndim+1]).to(device)
PBP = make_mlp([ndim*horizon] + [n_hid] * n_layers + [ndim]).to(device)
opt = T.optim.Adam([ {'params':model.parameters(), 'lr':0.001}, {'params':[Z], 'lr':0.1} ])
opt_PBP = T.optim.Adam([ {'params':PBP.parameters(), 'lr':0.001} ])
lr_sched = torch.optim.lr_scheduler.ExponentialLR(opt_PBP, gamma=0.999)


Z.requires_grad_()
losses = []
vars_ = []
zs = []
all_visited = []
first_visit = -1 * np.ones_like(true_dist)
l1log = []
nll_loss = torch.nn.NLLLoss()






for it in tqdm.trange(int(1000000.0/bs)):
    opt.zero_grad()
    
    z = T.zeros((bs,ndim), dtype=T.long).to(device)
    done = T.full((bs,), False, dtype=T.bool).to(device)
        
    action = None
    ll_diff = T.zeros((bs,)).to(device)
    ll_diff += Z
    
    
    i = 0
    z_save = []
    while T.any(~done):        
        pred = model(toin(z[~done]))
        edge_mask = T.cat([ (z[~done]==horizon-1).float(), T.zeros(((~done).sum(),1), device=device) ], 1)
        logits = (pred[...,:ndim+1] - 1000000000*edge_mask).log_softmax(1)

        exp_weight= 0.0
        temp = 1
        sample_ins_probs = (1-exp_weight)*(logits/temp).softmax(1) + exp_weight*(1-edge_mask) / (1-edge_mask+0.0000001).sum(1).unsqueeze(1)
        
        action = sample_ins_probs.multinomial(1)
        ll_diff[~done] += logits.gather(1, action).squeeze(1)
        terminate = (action==ndim).squeeze(1)
        
        for x in z[~done][terminate]:  
            state = (x.cpu()*(horizon**T.arange(ndim))).sum().item()
            if first_visit[state]<0: first_visit[state] = it
            all_visited.append(state)

        done[~done] |= terminate
        with T.no_grad():
            z[~done] = z[~done].scatter_add(1, action[~terminate], T.ones(action[~terminate].shape, dtype=T.long, device=device))
        
        
        if len(z[~done]) > 0:
            
            # Training of PBP
            opt_PBP.zero_grad()
            init_edge_mask = (z[~done]== 0).float()
            pred = ((PBP(toin(z[~done])))).log_softmax(1)
            nll_loss(pred, action[~terminate].reshape(-1)).backward()  
            opt_PBP.step()
        
            # Save for training GFN
            z_save.append((z.detach().clone(), 
                           done.detach().clone(), 
                           terminate.detach().clone(), 
                           action.detach().clone()))
            i += 1


    # Training of GFN
    for (z_, d_, t_, a_) in z_save:
        init_edge_mask = (z_[~d_]== 0).float()
        pred = ((PBP(toin(z_[~d_])))).log_softmax(1)
        ll_diff[~d_] -= pred.gather(1, a_[~t_]).squeeze(1).detach().clone()
    
    lens = z.sum(1)+1
    lr = log_reward(z.float())
    ll_diff -= lr

    loss = (ll_diff**2).sum()/ bs
    var = torch.var(ll_diff**2).item()
    loss.backward()

    opt.step()

    losses.append(loss.item())
    vars_.append(var)
    zs.append(Z.item())
    lr_sched.step()




    # Logging
    if it%100==0: 
        print('loss =', np.array(losses[-100:]).mean(), 'Z =', Z.item())
        emp_dist = np.bincount(all_visited[-200000:], minlength=len(true_dist)).astype(float)
        emp_dist /= emp_dist.sum()
        l1 = np.abs(true_dist-emp_dist).mean()
        print('L1 =', l1)
        l1log.append((len(all_visited), l1))
        to_log = {"L1": l1, "loss": np.array(losses[-100:]).mean(), "Variance": np.array(vars_[-100:]).mean()}
        
        
        wandb.log(to_log, step=it)
        if it % 1000 == 0:
            if ndim == 3:
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111, projection='3d')
                voxel_matrix = emp_dist.reshape(horizon,horizon,horizon)**0.45

                x, y, z = np.indices((horizon, horizon, horizon))
                colors = np.array([[0, 0, 1, voxel_matrix[xi, yi, zi]] if log_reward(torch.FloatTensor(np.array([xi,yi,zi])).float()).exp()<2.0 else [1, 1, 0, voxel_matrix[xi, yi, zi]**(0.45)]
                                   for xi, yi, zi in zip(x.ravel(), y.ravel(), z.ravel())])
                colors = colors.reshape(horizon,horizon,horizon,4)

                bins = horizon
                filled = np.zeros([horizon,horizon,horizon], dtype=np.bool_)
                for idx_x in range(bins):
                    for idx_y in range(bins):
                        for idx_z in range(bins):
                            if voxel_matrix[idx_x,idx_y,idx_z] > 0.000001:
                                filled[idx_x, idx_y, idx_z] = True
                                ax.voxels(filled=filled, 
                                          facecolors=colors.reshape(horizon,horizon,horizon,4))
                                filled[idx_x, idx_y, idx_z] = False


                ax.set_xlim(0, horizon)
                ax.set_ylim(0, horizon)
                ax.set_zlim(0, horizon)
                
                buf = io.BytesIO()
                fig.savefig(buf)
                buf.seek(0)
                pillow_image = Image.open(buf)
                plt.clf()
                plt.close()
                wandb.log({"P_T": wandb.Image(pillow_image)}, step=it)

