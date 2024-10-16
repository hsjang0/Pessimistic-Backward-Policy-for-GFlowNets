from itertools import chain
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
from torch_scatter import scatter, scatter_sum
import wandb
import torch.nn as nn

from .basegfn import BaseTBGFlowNet, tensor_to_np, unroll_trajs
from .. import network, utils

from torch.distributions import Categorical
from pathlib import Path

from .basegfn import unroll_trajs
from ..data import Experience
from ..utils import tensor_to_np, batch, pack, unpack
from ..network import make_mlp, StateFeaturizeWrap

class Empty(BaseTBGFlowNet):
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
  
  def train(self, batch):
    return


class TBGFN(BaseTBGFlowNet):
  
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
  
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
  
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    
  def train(self, batch):
    return self.train_tb(batch)

  def valid_loss(self, batch):
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    _, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    for i, exp in enumerate(batch):
      log_F_s[i, -1] = exp.logr.clone().detach()
    losses = (log_F_s[:,-1]+torch.sum(log_pb_actions.detach(),dim=1)
              - torch.sum(log_pf_actions,dim=1) - self.logZ).pow(2)#.sum(axis=1)
    #losses = torch.clamp(losses, max=20000)
    #mean_loss = torch.mean(losses)
    #self.count += 1
    return losses



class SubstructureGFN(BaseTBGFlowNet):
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)

  def train(self, batch):
    return self.train_substructure(batch)

  def train_substructure(self, batch, log = True):
    fwd_chain = self.batch_traj_fwd_logp(batch)
    back_chain = self.batch_traj_back_logp(batch)

    # 1. Obtain back policy loss
    logp_guide = torch.stack([exp.logp_guide for exp in batch])
    back_losses = torch.square(back_chain - logp_guide)
    back_losses = torch.clamp(back_losses, max=10**2)
    mean_back_loss = torch.mean(back_losses)

    # 2. Obtain TB loss with target: mix back_chain with logp_guide
    targets = []
    for i, exp in enumerate(batch):
      if exp.logp_guide is not None:
        w = self.args.target_mix_backpolicy_weight
        target = w * back_chain[i].detach() + (1 - w) * (exp.logp_guide + exp.logr)
      else:
        target = back_chain[i].detach()
      targets.append(target)
    targets = torch.stack(targets)

    tb_losses = torch.square(fwd_chain - targets)
    tb_losses = torch.clamp(tb_losses, max=10**2)
    loss_tb = torch.mean(tb_losses)

    # 1. Update back policy on back loss
    self.optimizer_back.zero_grad()
    loss_step1 = mean_back_loss
    loss_step1.backward()
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
    self.optimizer_back.step()
    if log:
      loss_step1 = tensor_to_np(loss_step1)
      print(f'Back training:', loss_step1)

    # 2. Update fwd policy on TB loss
    self.optimizer_fwdZ.zero_grad()
    loss_tb.backward()
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
    self.optimizer_fwdZ.step()
    self.clamp_logZ()
    if log:
      loss_tb = tensor_to_np(loss_tb)
      print(f'Fwd training:', loss_tb)

    if log:
      logZ = tensor_to_np(self.logZ)
      #print(f'{logZ=}')
      wandb.log({
        'Sub back loss': loss_step1,
        'Sub fwdZ loss': loss_tb,
        'Sub logZ': logZ,
      })
    return

  

class SubTBGFN(BaseTBGFlowNet):
  """ SubTB (lambda) """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    print("define state flow estimation")
    
  def init_subtb(self):
    r"""Precompute all possible subtrajectory indices that we will use for computing the loss:
    \sum_{m=1}^{T-1} \sum_{n=m+1}^T
        \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                    {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
    """
    self.subtb_max_len = self.mdp.forced_stop_len + 2
    ar = torch.arange(self.subtb_max_len, device=self.args.device)
    # This will contain a sequence of repeated ranges, e.g.
    # tidx[4] == tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    tidx = [torch.tril_indices(i, i, device=self.args.device)[1] for i in range(self.subtb_max_len)]
    # We need two sets of indices, the first are the source indices, the second the destination
    # indices. We precompute such indices for every possible trajectory length.

    # The source indices indicate where we index P_F and P_B, e.g. for m=3 and n=6 we'd need the
    # sequence [3,4,5]. We'll simply concatenate all sequences, for every m and n (because we're
    # computing \sum_{m=1}^{T-1} \sum_{n=m+1}^T), and get [0, 0,1, 0,1,2, ..., 3,4,5, ...].

    # The destination indices indicate the index of the subsequence the source indices correspond to.
    # This is used in the scatter sum to compute \log\prod_{i=m}^{n-1}. For the above example, we'd get
    # [0, 1,1, 2,2,2, ..., 17,17,17, ...]

    # And so with these indices, for example for m=0, n=3, the forward probability
    # of that subtrajectory gets computed as result[2] = P_F[0] + P_F[1] + P_F[2].

    self.precomp = [
        (
            torch.cat([i + tidx[T - i] for i in range(T)]),
            torch.cat(
                [ar[: T - i].repeat_interleave(ar[: T - i] + 1) + ar[T - i + 1 : T + 1].sum() for i in range(T)]
            ),
        )
        for T in range(1, self.subtb_max_len)
    ]
    self.lamda = self.args.lamda
    
  def train(self, batch):
    self.init_subtb()
    return self.train_subtb(batch)
  
  def train_subtb(self, batch, log = True):
    """ Step on trajectory balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_sub_trajectory_balance(batch)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    self.clamp_logZ()

    if log:
      batch_loss = tensor_to_np(batch_loss)
      print(f'TB training:', batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
    return
  
  def batch_loss_sub_trajectory_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    log_F_s[:, 0] = self.logZ.repeat(len(batch))
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = exp.logr.clone().detach()
      
    total_loss = torch.zeros(len(batch), device=self.args.device)
    ar = torch.arange(self.subtb_max_len)
    for i in range(len(batch)):
      # Luckily, we have a fixed terminal length
      idces, dests = self.precomp[-1]
      P_F_sums = scatter_sum(log_pf_actions[i, idces], dests)
      P_B_sums = scatter_sum(log_pb_actions[i, idces], dests)
      F_start = scatter_sum(log_F_s[i, idces], dests)
      F_end = scatter_sum(log_F_next_s[i, idces], dests)

      weight = torch.pow(self.lamda, torch.bincount(dests) - 1)
      total_loss[i] = (weight * (F_start - F_end + P_F_sums - P_B_sums).pow(2)).sum() / torch.sum(weight)
    losses = torch.clamp(total_loss, max=2000)
    mean_loss = torch.mean(losses)
    return mean_loss



class DBGFN(BaseTBGFlowNet):
  """ Detailed balance GFN """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    print("define state flow estimation")
    
  def train(self, batch):
    return self.train_db(batch)
  
  def train_db(self, batch, log = True):
    """ Step on detailed balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_detailed_balance(batch)
    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
      
    if log:
      batch_loss = tensor_to_np(batch_loss)
      print(f'TB training:', batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
      
    del batch_loss
    return
  
  def batch_loss_detailed_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = exp.logr.clone().detach()

    losses = (log_F_s + log_pf_actions - log_F_next_s - log_pb_actions).pow(2).sum(axis=1)
    losses = torch.clamp(losses, max=2000)
    mean_loss = torch.mean(losses)
    return mean_loss


class DBGFN_uniform(BaseTBGFlowNet):
  """ Detailed balance GFN """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    print("define state flow estimation")
    
  def train(self, batch):
    return self.train_db(batch)
  
  def train_db(self, batch, log = True):
    """ Step on detailed balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """


    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = exp.logr.clone().detach()

    losses = (log_F_s + log_pf_actions - log_F_next_s - log_pb_actions).pow(2).sum(axis=1)
    losses = torch.clamp(losses, max=2000)
    batch_loss = torch.mean(losses)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
      
    if log:
      batch_loss = tensor_to_np(batch_loss)
      print(f'TB training:', batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
      
    return
  
  def batch_loss_detailed_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = exp.logr.clone().detach()

    losses = (log_F_s + log_pf_actions - log_F_next_s - log_pb_actions).pow(2).sum(axis=1)
    losses = torch.clamp(losses, max=2000)
    mean_loss = torch.mean(losses)
    return mean_loss


  def back_logps_unique(self, batch):
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_dicts = []
    for state in batch:
      parents = self.mdp.get_unique_parents(state)
      logps = np.log([1/len(parents) for parent in parents])

      state_to_logp = {parent: logp for parent, logp in zip(parents, logps)}
      batch_dicts.append(state_to_logp)
    return batch_dicts if batched else batch_dicts[0]

  def back_sample(self, batch):
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_samples = []
    for state in batch:
      sample = np.random.choice(self.mdp.get_unique_parents(state))
      batch_samples.append(sample)
    return batch_samples if batched else batch_samples[0]




class GAFN(BaseTBGFlowNet):
  """ SubTB (lambda) """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.pR = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.pR.to(args.device)

    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.target = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.target.to(args.device)

    self.clip_grad_norm_params.append(self.logF.parameters())
    self.clip_grad_norm_params.append(self.pR.parameters())
    
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      },{
        'params': self.pR.parameters(),
        'lr': 1e-3
      }])
    self.count = 0
    self.optimizers.append(self.optimizer_logF)
    print("define state flow estimation")
    
  def train(self, batch):
    return self.train_tb(batch)
  
  def train_tb(self, batch, log = True):
    for opt in self.optimizers:
      opt.zero_grad()

    batch_loss = self.batch_loss_sub_trajectory_balance(batch)  
    batch_loss.backward(retain_graph = True)
    
    for param_set in self.clip_grad_norm_params:
      # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    self.clamp_logZ()

    if log:
      batch_loss = tensor_to_np(batch_loss)
      print(f'TB training:', batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
    return


  def batch_loss_sub_trajectory_balance(self, batch):
    reward_est, log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    
    losses_est = torch.sum(torch.mean((reward_est), dim = -1))
    losses_est = losses_est / len(batch)
    losses_est = torch.clamp(losses_est, max=2000)
    losses_est.backward(retain_graph = True)
    log_F_s[:, 0] = self.logZ.repeat(len(batch))
    
    
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = 0
    
    total_loss = torch.zeros(len(batch), device=self.args.device)
    reward_est = reward_est.detach()
    
    reward_est_2 = torch.zeros(reward_est.shape, device=self.args.device)
    reward_est_2 = reward_est
    reward_est_2[:,1:] = reward_est_2[:,1:] - reward_est[:,:-1]
    
    for i, exp in enumerate(batch):
      P_F_sums = torch.sum(log_pf_actions[i])
      P_B_sums = torch.sum(log_pb_actions[i])
      riss = torch.relu(torch.sum(reward_est_2[i]))
      total_loss[i] = ((self.logZ - torch.log(torch.exp(exp.logr)+0.01*riss) + P_F_sums - P_B_sums).pow(2))
      
    losses = torch.clamp(total_loss, max=5000)
    mean_loss = torch.mean(losses)
  
    self.count += 1
    return mean_loss #+ losses_est
  
  
  def batch_traj_fwd_logp_unroll(self, batch):
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)

    
    in_reward = fwd_states
    reward_est = torch.norm(self.pR(in_reward) - self.target(in_reward).detach(), dim=-1, p=2)
    #print(reward_est.shape)
    states_to_logfs = self.fwd_logfs_unique(fwd_states)
    states_to_logps = self.fwd_logps_unique(fwd_states)
    fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_logps, back_states)]

    reward_est_2d = torch.zeros((len(batch), self.mdp.forced_stop_len+1)).to(device=self.args.device)
    log_F_s = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    log_pf_actions = torch.zeros((len(batch), self.mdp.forced_stop_len + 1)).to(device=self.args.device)
    for traj_idx, (start, end) in unroll_idxs.items():
      for i, j in enumerate(range(start, end)):
        reward_est_2d[traj_idx][i] = reward_est[j]
        log_F_s[traj_idx][i] = states_to_logfs[j]
        log_pf_actions[traj_idx][i] = fwd_logp_chosen[j]
    
    #reward_est_2d[:,-1] = 0
    return reward_est_2d, log_F_s, log_pf_actions







class TB_uniform(BaseTBGFlowNet):
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
  
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
  
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    

  def train(self, batch):
    return self.train_tb(batch)

  def back_logps_unique(self, batch):
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_dicts = []
    for state in batch:
      parents = self.mdp.get_unique_parents(state)
      logps = np.log([1/len(parents) for parent in parents])

      state_to_logp = {parent: logp for parent, logp in zip(parents, logps)}
      batch_dicts.append(state_to_logp)
    return batch_dicts if batched else batch_dicts[0]

  def back_sample(self, batch):
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_samples = []
    for state in batch:
      sample = np.random.choice(self.mdp.get_unique_parents(state))
      batch_samples.append(sample)
    return batch_samples if batched else batch_samples[0]


  def valid_loss(self, batch):
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    _, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    for i, exp in enumerate(batch):
      log_F_s[i, -1] = exp.logr.clone().detach()
    losses = (log_F_s[:,-1]+torch.sum(log_pb_actions.detach(),dim=1)
              - torch.sum(log_pf_actions,dim=1) - self.logZ).pow(2)#.sum(axis=1)
    return losses



class real_MaxEntGFN(BaseTBGFlowNet):
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
  
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
  
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    

  def train(self, batch):
    return self.train_tb(batch)

  def back_logps_unique(self, batch):
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_dicts = []
    for state in batch:
      parents = self.mdp.get_unique_parents(state)
      logps = np.log([float(parent.count_comb()/float(state.count_comb()))
                      for parent in parents])

      state_to_logp = {parent: logp for parent, logp in zip(parents, logps)}
      batch_dicts.append(state_to_logp)
    return batch_dicts if batched else batch_dicts[0]

  def back_sample(self, batch):
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_samples = []
    for state in batch:
      sample = np.random.choice(self.mdp.get_unique_parents(state))
      batch_samples.append(sample)
    return batch_samples if batched else batch_samples[0]


  def valid_loss(self, batch):
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    _, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    for i, exp in enumerate(batch):
      log_F_s[i, -1] = exp.logr.clone().detach()
    losses = (log_F_s[:,-1]+torch.sum(log_pb_actions.detach(),dim=1)
              - torch.sum(log_pf_actions,dim=1) - self.logZ).pow(2)#.sum(axis=1)
    #losses = torch.clamp(losses, max=20000)
    #mean_loss = torch.mean(losses)
    #self.count += 1
    return losses






class SubTBGFN_uniform(BaseTBGFlowNet):
  """ SubTB (lambda) """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    print("define state flow estimation")


  def back_logps_unique(self, batch):
    """ Uniform distribution over parents.

        Other idea - just call parent back_logps_unique, then replace
        predicted logps.
        see policy.py : logps_unique(batch)

        Output logps of unique children/parents.

        Typical logic flow (example for getting children)
         Call network on state - returns high-dim actions
        2. Translate actions into list of states - not unique
        3. Filter invalid child states
        4. Reduce states to unique, using hash property of states.
           Need to add predicted probabilities.
        5. Normalize probs to sum to 1

        Input: List of [State], n items
        Returns
        -------
        logps: n-length List of torch.tensor of logp.
            Each tensor can have different length.
        states: List of List of [State]; must be unique.
            Each list can have different length.
    """
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_dicts = []
    for state in batch:
      parents = self.mdp.get_unique_parents(state)
      logps = np.log([1/len(parents) for parent in parents])

      state_to_logp = {parent: logp for parent, logp in zip(parents, logps)}
      batch_dicts.append(state_to_logp)
    return batch_dicts if batched else batch_dicts[0]


  def init_subtb(self):
    r"""Precompute all possible subtrajectory indices that we will use for computing the loss:
    \sum_{m=1}^{T-1} \sum_{n=m+1}^T
        \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                    {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
    """
    self.subtb_max_len = self.mdp.forced_stop_len + 2
    ar = torch.arange(self.subtb_max_len, device=self.args.device)
    # This will contain a sequence of repeated ranges, e.g.
    # tidx[4] == tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    tidx = [torch.tril_indices(i, i, device=self.args.device)[1] for i in range(self.subtb_max_len)]
    # We need two sets of indices, the first are the source indices, the second the destination
    # indices. We precompute such indices for every possible trajectory length.

    # The source indices indicate where we index P_F and P_B, e.g. for m=3 and n=6 we'd need the
    # sequence [3,4,5]. We'll simply concatenate all sequences, for every m and n (because we're
    # computing \sum_{m=1}^{T-1} \sum_{n=m+1}^T), and get [0, 0,1, 0,1,2, ..., 3,4,5, ...].

    # The destination indices indicate the index of the subsequence the source indices correspond to.
    # This is used in the scatter sum to compute \log\prod_{i=m}^{n-1}. For the above example, we'd get
    # [0, 1,1, 2,2,2, ..., 17,17,17, ...]

    # And so with these indices, for example for m=0, n=3, the forward probability
    # of that subtrajectory gets computed as result[2] = P_F[0] + P_F[1] + P_F[2].

    self.precomp = [
        (
            torch.cat([i + tidx[T - i] for i in range(T)]),
            torch.cat(
                [ar[: T - i].repeat_interleave(ar[: T - i] + 1) + ar[T - i + 1 : T + 1].sum() for i in range(T)]
            ),
        )
        for T in range(1, self.subtb_max_len)
    ]
    self.lamda = self.args.lamda
    
  def train(self, batch):
    self.init_subtb()
    return self.train_subtb(batch)
  
  def train_subtb(self, batch, log = True):
    """ Step on trajectory balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_sub_trajectory_balance(batch)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    self.clamp_logZ()

    if log:
      batch_loss = tensor_to_np(batch_loss)
      print(f'TB training:', batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
    return
  
  def batch_loss_sub_trajectory_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    log_F_s[:, 0] = self.logZ.repeat(len(batch))
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = exp.logr.clone().detach()
      
    total_loss = torch.zeros(len(batch), device=self.args.device)
    ar = torch.arange(self.subtb_max_len)
    for i in range(len(batch)):
      # Luckily, we have a fixed terminal length
      idces, dests = self.precomp[-1]
      P_F_sums = scatter_sum(log_pf_actions[i, idces], dests)
      P_B_sums = scatter_sum(log_pb_actions[i, idces], dests)
      F_start = scatter_sum(log_F_s[i, idces], dests)
      F_end = scatter_sum(log_F_next_s[i, idces], dests)

      weight = torch.pow(self.lamda, torch.bincount(dests) - 1)
      total_loss[i] = (weight * (F_start - F_end + P_F_sums - P_B_sums).pow(2)).sum() / torch.sum(weight)
    losses = torch.clamp(total_loss, max=2000)
    mean_loss = torch.mean(losses)
    return mean_loss






class PPO():
    """
        Proximal Policy Gradient
        Actor: SSR style
        Critic: SA style
    """
    
    def __init__(self, args, mdp, actor):
        self.args = args
        self.mdp = mdp
        self.actor = actor
        
        self.policy = actor.policy_fwd
        self.policy_back = actor.policy_back # not used
        
        hid_dim = self.args.sa_hid_dim
        n_layers = self.args.sa_n_layers
        net = make_mlp(
            [self.actor.ft_dim] + \
            [hid_dim] * n_layers + \
            [1]
        )
        self.critic = StateFeaturizeWrap(net, self.actor.featurize)
        self.critic.to(args.device)
        
        self.nets = [self.policy, self.critic]
        for net in self.nets:
            net.to(self.args.device)
            
        self.clip_grad_norm_params = [self.policy.parameters(),
                                      self.critic.parameters()]
        
        self.optimizer = torch.optim.Adam([
            {
                'params': self.policy.parameters(),
                'lr': args.lr_policy
            }, {
                'params': self.critic.parameters(),
                'lr': args.lr_critic
            }
        ])
    
    def fwd_sample(self, batch, epsilon=0.0):
        return self.policy.sample(batch, epsilon=epsilon)
    
    def fwd_logps_unique(self, batch):
        return self.policy.logps_unique(batch)
    
    def batch_fwd_sample(self, n, epsilon=0.0, uniform=False):
        """ Batch samples dataset with n items.

            Parameters
            ----------
            n: int, size of dataset.
            epsilon: Chance in [0, 1] of uniformly sampling a unique child.
            uniform: If true, overrides epsilon to 1.0
            unique: bool, whether all samples should be unique

            Returns
            -------
            dataset: List of [Experience]
        """
        print('Sampling dataset ...')
        if uniform:
            print('Using uniform forward policy on unique children ...')
            epsilon = 1.0
        incomplete_trajs = [[self.mdp.root()] for _ in range(n)]
        complete_trajs = []
        logps_trajs = [[] for _ in range(n)]
        while len(incomplete_trajs) > 0:
            inp = [t[-1] for t in incomplete_trajs]
            samples = self.fwd_sample(inp, epsilon=epsilon)
            logps = self.fwd_logps_unique(inp)
            for i, (logp, sample) in enumerate(zip(logps, samples)):
                incomplete_trajs[i].append(sample)
                logps_trajs[i].append(logp[sample].cpu().detach())
        
            # Remove complete trajs that hit leaf
            temp_incomplete = []
            for t in incomplete_trajs:
                if not t[-1].is_leaf:
                    temp_incomplete.append(t)
                else:
                    complete_trajs.append(t)
            incomplete_trajs = temp_incomplete

        # convert trajs to exps
        list_exps = []
        for traj, logps_traj in zip(complete_trajs, logps_trajs):
            x = traj[-1]
            r = self.mdp.reward(x)
            # prevent NaN
            exp = Experience(traj=traj, x=x, r=r,
                logr=torch.nan_to_num(torch.log(torch.tensor(r, dtype=torch.float32)).to(device=self.args.device), neginf=-100.0),
                logp_guide=logps_traj)
            list_exps.append(exp)
        return list_exps
        
    def train(self, batch, log = True):
        batch_loss = self.batch_loss_ppo(batch)
        
        self.optimizer.zero_grad()
        batch_loss.backward()
        
        for param_set in self.clip_grad_norm_params:
            # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
        self.optimizer.step()
        
        if log:
            batch_loss = tensor_to_np(batch_loss)
            print(f'PPO training:', batch_loss)
            wandb.log({'PPO loss': batch_loss})
            return
        
    def batch_loss_ppo(self, batch):
        trajs = [exp.traj for exp in batch]
        old_fwd_logp_chosen = [exp.logp_guide for exp in batch]
        fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)
        
        states_to_logps = self.fwd_logps_unique(fwd_states)
        fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_logps, back_states)]
        old_log_probs = torch.zeros((len(fwd_logp_chosen), )).to(self.args.device)
        log_probs = torch.zeros((len(fwd_logp_chosen), )).to(self.args.device)
        for i, (old_log_prob, log_prob) in enumerate(zip(old_fwd_logp_chosen, fwd_logp_chosen)):
            old_log_probs[i] = old_log_prob[i % (self.mdp.forced_stop_len + 1)].to(self.args.device)
            log_probs[i] = log_prob
            
        old_log_probs = self.clip_policy_logits(old_log_probs)
        old_log_probs = torch.nan_to_num(old_log_probs, neginf=self.args.clip_policy_logit_min)
        
        log_probs = self.clip_policy_logits(log_probs)
        log_probs = torch.nan_to_num(log_probs, neginf=self.args.clip_policy_logit_min)

        V = self.critic(fwd_states)
        # The return is the terminal reward everywhere, we're using gamma==1
        G = torch.FloatTensor([exp.r for exp in batch]).repeat_interleave(self.mdp.forced_stop_len + 1).to(self.args.device)
        A = G - V
        
        V_loss = A.pow(2)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * A
        pol_objective = torch.min(surr1, surr2)
        entropy = torch.zeros((len(fwd_logp_chosen), )).to(self.args.device)
        for i, s2lp in enumerate(states_to_logps):
            for state, logp in s2lp.items():
                entropy[i] = -torch.sum(torch.exp(logp) * logp)
        pol_objective = pol_objective + self.args.entropy_coef * entropy
        pol_loss = -pol_objective
        
        loss = V_loss + pol_loss
        loss = torch.clamp(loss, max=5000)
        mean_loss = torch.mean(loss)
        return mean_loss
    
    def save_params(self, file):
        print('Saving checkpoint model ...')
        Path('/'.join(file.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        #torch.save({
        #    'policy':   self.policy.state_dict(),
        #    'critic':  self.critic.state_dict(),
        #    }, file)
        return

    def load_for_eval_from_checkpoint(self, file):
        print(f'Loading checkpoint model ...')
        checkpoint = torch.load(file)
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        for net in self.nets:
            net.eval()
        return

    def clip_policy_logits(self, scores):
        return torch.clip(scores, min=self.args.clip_policy_logit_min,
                                  max=self.args.clip_policy_logit_max)


from gflownet.MDPs.seqpamdp import SeqPAState, SeqPAActionType, SeqPAAction







class PBP(BaseTBGFlowNet):
  def __init__(self, args, mdp, actor):
    self.args = args
    self.count = 0
    self.mdp = mdp
    self.actor = actor

    self.policy_fwd = actor.policy_fwd
    self.policy_back = actor.policy_back

    self.logZ = torch.nn.Parameter(torch.tensor([5.], device=self.args.device))
    self.logZ_lower = 0

    self.nets = [self.policy_fwd, self.policy_back]
    for net in self.nets:
      net.to(args.device)

    self.clip_grad_norm_params = [self.policy_fwd.parameters(),
                                  self.policy_back.parameters()]

    self.optimizer_back = torch.optim.Adam([
        {
          'params': self.policy_back.parameters(),
          'lr': 1e-3
        }])
    self.optimizer_fwdZ = torch.optim.Adam([
        {
          'params': self.policy_fwd.parameters(),     
          'lr': 1e-4
        }, {
          'params': self.logZ, 
          'lr': args.lr_z
        }])
    self.optimizers = [self.optimizer_fwdZ, self.optimizer_back]
    
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.actor.ft_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
  

  def train(self, batch):
    return self.train_tb(batch)


  
  def train_pbp(self, batch):

    for opt in self.optimizers:
      opt.zero_grad()
      
    _, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    losses_est = torch.sum(-log_pb_actions)
    losses_est = losses_est / len(batch)
    losses_est = torch.clamp(losses_est, max=10000)
    losses_est.backward()
  
    for opt in self.optimizers:
      opt.step()    
    return
  
  
  def train_tb(self, batch, log = True):
    batch_loss = self.batch_loss_trajectory_balance(batch)
    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
    #  # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()

    if log:
      batch_loss = tensor_to_np(batch_loss)
      print(f'TB training:', batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
    return
  
  
  def batch_loss_trajectory_balance(self, batch):
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    _, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    for i, exp in enumerate(batch):
      log_F_s[i, -1] = exp.logr.clone().detach()
    losses = (log_F_s[:,-1]+torch.sum(log_pb_actions.detach(),dim=1)
              - torch.sum(log_pf_actions,dim=1) - self.logZ).pow(2)#.sum(axis=1)
    losses = torch.clamp(losses, max=20000)
    mean_loss = torch.mean(losses)
    self.count += 1
    return mean_loss


  def valid_loss(self, batch):
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    _, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    for i, exp in enumerate(batch):
      log_F_s[i, -1] = exp.logr.clone().detach()
    losses = (log_F_s[:,-1]+torch.sum(log_pb_actions.detach(),dim=1)
              - torch.sum(log_pf_actions,dim=1) - self.logZ).pow(2)#.sum(axis=1)
    return losses










class A2C():
    """
        Advantage Actor Critic
        Actor: SSR style
        Critic: SA style
    """
    
    def __init__(self, args, mdp, actor):
        self.args = args
        self.mdp = mdp
        self.actor = actor
        
        self.policy = actor.policy_fwd
        self.policy_back = actor.policy_back # not used
        
        hid_dim = self.args.sa_hid_dim
        n_layers = self.args.sa_n_layers
        net = make_mlp(
            [self.actor.ft_dim] + \
            [hid_dim] * n_layers + \
            [1]
        )
        self.critic = StateFeaturizeWrap(net, self.actor.featurize)
        self.critic.to(args.device)
        
        self.nets = [self.policy, self.critic]
        for net in self.nets:
            net.to(self.args.device)
            
        self.clip_grad_norm_params = [self.policy.parameters(),
                                      self.critic.parameters()]
        
        self.optimizer = torch.optim.Adam([
            {
                'params': self.policy.parameters(),
                'lr': args.lr_policy
            }, {
                'params': self.critic.parameters(),
                'lr': args.lr_critic
            }
        ])
    
    def fwd_sample(self, batch, epsilon=0.0):
        return self.policy.sample(batch, epsilon=epsilon)
    
    def fwd_logps_unique(self, batch):
        return self.policy.logps_unique(batch)
    
    def batch_fwd_sample(self, n, epsilon=0.0, uniform=False):
        """ Batch samples dataset with n items.

            Parameters
            ----------
            n: int, size of dataset.
            epsilon: Chance in [0, 1] of uniformly sampling a unique child.
            uniform: If true, overrides epsilon to 1.0
            unique: bool, whether all samples should be unique

            Returns
            -------
            dataset: List of [Experience]
        """
        print('Sampling dataset ...')
        if uniform:
            print('Using uniform forward policy on unique children ...')
            epsilon = 1.0
        incomplete_trajs = [[self.mdp.root()] for _ in range(n)]
        complete_trajs = []
        while len(incomplete_trajs) > 0:
            inp = [t[-1] for t in incomplete_trajs]
            samples = self.fwd_sample(inp, epsilon=epsilon)
            for i, sample in enumerate(samples):
                incomplete_trajs[i].append(sample)
        
            # Remove complete trajs that hit leaf
            temp_incomplete = []
            for t in incomplete_trajs:
                if not t[-1].is_leaf:
                    temp_incomplete.append(t)
                else:
                    complete_trajs.append(t)
            incomplete_trajs = temp_incomplete

        # convert trajs to exps
        list_exps = []
        for traj in complete_trajs:
            x = traj[-1]
            r = self.mdp.reward(x)
            # prevent NaN
            exp = Experience(traj=traj, x=x, r=r,
                logr=torch.nan_to_num(torch.log(torch.tensor(r, dtype=torch.float32)).to(device=self.args.device), neginf=-100.0))
            list_exps.append(exp)
        return list_exps
        
    def train(self, batch, log = True):
        batch_loss = self.batch_loss_advantage_actor_critic(batch)
        
        self.optimizer.zero_grad()
        batch_loss.backward()
        
        for param_set in self.clip_grad_norm_params:
            # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
        self.optimizer.step()
        
        if log:
            batch_loss = tensor_to_np(batch_loss)
            print(f'A2C training:', batch_loss)
            wandb.log({'A2C loss': batch_loss})
            return
        
    def batch_loss_advantage_actor_critic(self, batch):
        trajs = [exp.traj for exp in batch]
        fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)
        
        states_to_logps = self.fwd_logps_unique(fwd_states)
        fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_logps, back_states)]
        log_probs = torch.zeros((len(fwd_logp_chosen), )).to(self.args.device)
        for i, log_prob in enumerate(fwd_logp_chosen):
            log_probs[i] = log_prob
        log_probs = self.clip_policy_logits(log_probs)
        log_probs = torch.nan_to_num(log_probs, neginf=self.args.clip_policy_logit_min)
        
        V = self.critic(fwd_states)
        # The return is the terminal reward everywhere, we're using gamma==1
        G = torch.FloatTensor([exp.r for exp in batch]).repeat_interleave(self.mdp.forced_stop_len + 1).to(self.args.device)
        A = G - V
        
        V_loss = A.pow(2)
        pol_objective = (log_probs * A.detach())
        entropy = torch.zeros((len(fwd_logp_chosen), )).to(self.args.device)
        for i, s2lp in enumerate(states_to_logps):
            for state, logp in s2lp.items():
                entropy[i] = -torch.sum(torch.exp(logp) * logp)
        pol_objective = pol_objective + self.args.entropy_coef * entropy
        pol_loss = -pol_objective
        
        loss = V_loss + pol_loss
        loss = torch.clamp(loss, max=5000)
        mean_loss = torch.mean(loss)
        return mean_loss
    
    def save_params(self, file):
        print('Saving checkpoint model ...')
        Path('/'.join(file.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy':   self.policy.state_dict(),
            'critic':  self.critic.state_dict(),
            }, file)
        return

    def load_for_eval_from_checkpoint(self, file):
        print(f'Loading checkpoint model ...')
        checkpoint = torch.load(file)
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        for net in self.nets:
            net.eval()
        return

    def clip_policy_logits(self, scores):
        return torch.clip(scores, min=self.args.clip_policy_logit_min,
                                  max=self.args.clip_policy_logit_max)
        
        






def make_model(args, mdp, actor):
  """ Constructs MaxEnt / TB / Sub GFN. """
  if args.model == 'tb_uniform':
    model = TB_uniform(args, mdp, actor)
  elif args.model == 'maxent':
    model = real_MaxEntGFN(args, mdp, actor)
  elif args.model == 'a2c':
    model = A2C(args, mdp, actor)
  elif args.model == 'tb':
    model = TBGFN(args, mdp, actor)
  elif args.model == 'gafn':
    model = GAFN(args, mdp, actor)
  elif args.model == 'sub':
    model = SubstructureGFN(args, mdp, actor)
  elif args.model == "subtb":
    model = SubTBGFN(args, mdp, actor)
  elif args.model == 'db':
    model = DBGFN(args, mdp, actor)
  elif args.model == 'db_uniform':
    model = DBGFN_uniform(args, mdp, actor)
  elif args.model == 'subtb_uniform':
    model = SubTBGFN_uniform(args, mdp, actor)
  elif args.model == 'pbp':
    model = PBP(args, mdp, actor)
  elif args.model == 'random':
    args.explore_epsilon = 1.0
    args.num_offline_batches_per_round = 0
    model = Empty(args, mdp, actor)
  return model







