import random, time
import pickle
import numpy as np
import torch
import wandb
from tqdm import tqdm
# import ray
import gc
import random
from . import guide
from .data import Experience


class Trainer:
  def __init__(self, args, model, mdp, actor, monitor):
    self.args = args
    self.model = model
    self.mdp = mdp
    self.actor = actor
    self.monitor = monitor

  def learn(self, *args, **kwargs):
    if self.args.model == 'sub':
      print(f'Learning with ray guide workers ...')
      self.learn_with_ray_workers(*args, **kwargs)
    else:
      print(f'Learning without guide workers ...')
      self.learn_default(*args, **kwargs)

  def handle_init_dataset(self, initial_XtoR):
    if initial_XtoR:
      print(f'Using initial dataset of size {len(initial_XtoR)}. \
              Skipping first online round ...')
      if self.args.init_logz:
        self.model.init_logz(np.log(sum(initial_XtoR.values())))
    else:
      print(f'No initial dataset used')
    return

  """
    Training
  """
  def learn_default(self, initial_XtoR=None, ground_truth=None):
    """ Main learning training loop.
        Each learning round:
          Each online batch:
            sample a new dataset using exploration policy.
          Each offline batch:
            resample batch from full historical dataset
        Monitor exploration - judge modes with monitor_explore callable.

        To learn on fixed dataset only: Set 0 online batches per round,
        and provide initial dataset.

        dataset = List of [Experience]
    """
    #allall = ground_truth
    munge = lambda x: ''.join([str(c) for c in list(x)])
    if len(self.mdp.modes) == 3:
      ground_mode = list(self.mdp.modes[0])
      if isinstance(ground_mode[0], str):
        ground_mode = [self.mdp.state(munge(x), is_leaf=True) for x in ground_mode]
    else:
      ground_mode = list(self.mdp.modes)
    
    
    # Initialization
    allXtoR = initial_XtoR if initial_XtoR else dict()
    self.handle_init_dataset(initial_XtoR)
    num_online = self.args.num_online_batches_per_round
    num_offline = self.args.num_offline_batches_per_round
    online_bsize = self.args.num_samples_per_online_batch
    offline_bsize = self.args.num_samples_per_offline_batch
    print(f'Starting active learning. \
            Each round: num_online={num_online}, num_offline={num_offline}')
    total_samples = []
    buffer_for_PBP = []    
    
    
    # Training
    for round_num in tqdm(range(self.args.num_active_learning_rounds)):
      print(f'Starting learning round {round_num+1} / {self.args.num_active_learning_rounds} ...')
      if not initial_XtoR or round_num > 0:
        for _ in range(num_online):
          
               
          # Sampling trajectories
          with torch.no_grad():
            explore_data = self.model.batch_fwd_sample(online_bsize,
                  epsilon=self.args.explore_epsilon)
            buffer_for_PBP.append(explore_data)
            buffer_for_PBP = buffer_for_PBP[-self.args.buffer_size:] # buffer size (for pessimistic training) is 20
          for exp in explore_data:
            if exp.x not in allXtoR:
              allXtoR[exp.x] = exp.r                  
          total_samples.extend(explore_data)

          
          # Pessimistic training of backward policy
          for step_num in range(self.args.alg_N):
            if self.args.model in ['pbp']:
              self.model.train_pbp(buffer_for_PBP[-random.randint(0,len(buffer_for_PBP)-1)])
        

          # Training of GFlowNets
          if self.args.model not in ['a2c']:
            for step_num in range(self.args.num_steps_per_batch):
              self.model.train(explore_data)
                    
                    
      # Training code for A2C
      for _ in range(num_offline):
        if self.args.model == "a2c":
          # As A2C is off-policy algorithm, we double offline training steps
          offline_dataset = random.choices(total_samples, k=offline_bsize)
          for step_num in range(self.args.num_steps_per_batch * 2):
            self.model.train(offline_dataset)


      # Offline training
      if self.args.model not in ['a2c']:
        for _ in range(num_offline):
          with torch.no_grad():
            offline_xs = self.select_offline_xs(allXtoR, offline_bsize)
            offline_dataset = self.offline_PB_traj_sample(offline_xs, allXtoR)
            buffer_for_PBP.append(offline_dataset)
            buffer_for_PBP = buffer_for_PBP[-self.args.buffer_size:] # buffer size (for pessimistic training) is 20
            
            
          # Pessimistic training of backward policy
          for step_num in range(self.args.alg_N):
            if self.args.model in ['pbp']:
              self.model.train_pbp(buffer_for_PBP[-random.randint(0,len(buffer_for_PBP)-1)])
          
          
          # Training of GFlowNets
          for step_num in range(self.args.num_steps_per_batch):
            self.model.train(offline_dataset)
          
      
      # Evaluation
      if round_num % 10 == 0:
        with torch.no_grad():
          explore_data = self.model.batch_fwd_sample(128,
              epsilon=0)
          new_test = [
              Experience(x=dd.x, r=dd.r,
                        logr=np.log(dd.r)
                        )
              for dd in explore_data
            ]
          explore_data.clear()
          self.monitor.log_samples(round_num, new_test)
          self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)


      gc.collect()
      torch.cuda.empty_cache()
      if round_num and round_num % self.args.save_every_x_active_rounds == 0:
        self.model.save_params(self.args.saved_models_dir + \
                               self.args.run_name + "/" + f'{wandb.run.id}_round_{round_num}.pth')
        with open(self.args.saved_models_dir + \
                  self.args.run_name + "/" + f"{wandb.run.id}_round_{round_num}_sample.pkl", "wb") as f:
          pickle.dump(total_samples, f)
    print('Finished training.')
    return







  """
    Learn with ray workers
  """
  def learn_with_ray_workers(self, initial_XtoR=None):
    """ Guided trajectory balance - ray workers compute guide trajectories.
    """
    # Ray init
    # ray.init(num_cpus = self.args.num_guide_workers)
    guidemanager = guide.RayManager(self.args, self.mdp)

    allXtoR = initial_XtoR if initial_XtoR else dict()
    guidemanager.update_allXtoR(allXtoR)
    self.handle_init_dataset(initial_XtoR)

    num_online = self.args.num_online_batches_per_round
    num_offline = self.args.num_offline_batches_per_round
    online_bsize = self.args.num_samples_per_online_batch
    offline_bsize = self.args.num_samples_per_offline_batch
    monitor_fast_every = self.args.monitor_fast_every
    monitor_num_samples = self.args.monitor_num_samples
    # print(f'Starting active learning. \
    #         Each round: {num_online=}, {num_offline=}')
    print(f'Starting active learning. \
            Each round: num_online={num_online}, num_offline={num_offline}')

    total_samples = []
    for round_num in tqdm(range(self.args.num_active_learning_rounds)):
      print(f'Starting learning round {round_num+1}/{self.args.num_active_learning_rounds} ...')
      
      # 1. Sample online x with explore policy
      for _ in range(num_online):
        print(f'Sampling new x ...')
        
        if self.args.mcmc:
          with torch.no_grad():
            explore_data = self.model.batch_fwd_sample_mcmc(online_bsize,
              epsilon=self.args.explore_epsilon, k=self.args.k)
        else:
          with torch.no_grad():
            explore_data = self.model.batch_fwd_sample(online_bsize,
                epsilon=self.args.explore_epsilon)

        online_xs = [exp.x for exp in explore_data]
        for exp in explore_data:
          if exp.x not in allXtoR:
            allXtoR[exp.x] = exp.r
        guidemanager.update_allXtoR(allXtoR)

        # 2. Submit online jobs - get guide traj for x
        guidemanager.submit_online_jobs(online_xs)

      # 2a. Submit offline jobs
      for _ in range(num_offline):
        offline_xs = self.select_offline_xs(allXtoR, offline_bsize)

        # Submit offline jobs
        if self.args.offline_style == 'guide_resamples_traj':
          guidemanager.submit_online_jobs(offline_xs)
        if self.args.offline_style == 'guide_scores_back_policy_traj':
          print(f'Sampling offline trajectories with back policy ...')
          with torch.no_grad():
            offline_trajs = self.model.batch_back_sample(offline_xs)
          guidemanager.submit_offline_jobs(offline_trajs)

      # 4. Train if possible
      for _ in range(num_online + num_offline):
        batch = guidemanager.get_results(batch_size=online_bsize)
        if batch is not None:
          print(f'Training ...')
          for step_num in range(self.args.num_steps_per_batch):
            self.model.train(batch)



      if round_num % 10 == 0:
        with torch.no_grad():
          explore_data = self.model.batch_fwd_sample(128,
              epsilon=0)
          new_test = [
              Experience(x=dd.x, r=dd.r,
                        logr=np.log(dd.r)
                        )
              for dd in explore_data
            ]
          total_samples.extend(new_test)
          explore_data.clear()
          self.monitor.log_samples(round_num, new_test)
          self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)
          
          
      
    
      if round_num and round_num % self.args.save_every_x_active_rounds == 0:
        self.model.save_params(self.args.saved_models_dir + \
                               self.args.run_name + "/" + f'{wandb.run.id}_round_{round_num}.pth')
        with open(self.args.saved_models_dir + \
                  self.args.run_name + "/" + f"{wandb.run.id}_round_{round_num}_sample.pkl", "wb") as f:
          pickle.dump(total_samples, f)
          
    print('Finished training.')
    #self.model.save_params(self.args.saved_models_dir + \
    #                       self.args.run_name + "/" + 'final.pth')
    #self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)
    return

  """
    Offline training
  """
  def select_offline_xs(self, allXtoR, batch_size):
    select = self.args.get('offline_select', 'prt')
    if select == 'prt':
      return self.__biased_sample_xs(allXtoR, batch_size)
    elif select == 'random':
      return self.__random_sample_xs(allXtoR, batch_size)

  def __biased_sample_xs(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State].
        Draws 50% from top 10% of rewards, and 50% from bottom 90%. 
    """
    if len(allXtoR) < 10:
      return []

    rewards = np.array(list(allXtoR.values()))
    threshold = np.percentile(rewards, 90)
    top_xs = [x for x, r in allXtoR.items() if r >= threshold]
    bottom_xs = [x for x, r in allXtoR.items() if r <= threshold]
    sampled_xs = random.choices(top_xs, k=batch_size // 2) + \
                 random.choices(bottom_xs, k=batch_size // 2)
    return sampled_xs

  def __biased_sample_xs_for_valid(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State].
        Draws 50% from top 10% of rewards, and 50% from bottom 90%. 
    """
    if len(allXtoR) < 10:
      return []
    rewards = np.array(list(allXtoR.values()))
    threshold = np.percentile(rewards, 99)
    top_xs = [x for x, r in allXtoR.items() if r >= threshold]
    sampled_xs = random.choices(top_xs, k=batch_size)
    return sampled_xs

  def __random_sample_xs(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State]. """
    return random.choices(list(allXtoR.keys()), k=batch_size)

  def __valid_sample_xs(self, ground_mode, batch_size):
    """ Select xs for offline training. Returns List of [State]. """
    return random.choices(ground_mode, k=batch_size)

  def valid_PB_traj_sample(self, offline_xs, allXtoR):
    """ Sample trajectories for x using P_B, for offline training with TB.
        Returns List of [Experience].
    """
    
    offline_rs = [self.mdp.reward(x) for x in offline_xs]


    # Not subgfn: sample trajectories from backward policy
    with torch.no_grad():
      offline_trajs = self.model.batch_back_sample(offline_xs)

    
    offline_trajs = [
      Experience(traj=traj, x=x, r=r,
                logr=torch.log(torch.tensor(r, dtype=torch.float32,device=self.args.device))
                )
      for traj, x, r in zip(offline_trajs, offline_xs, offline_rs)
    ]
    
    return offline_trajs




  def offline_PB_traj_sample(self, offline_xs, allXtoR):
    """ Sample trajectories for x using P_B, for offline training with TB.
        Returns List of [Experience].
    """

    offline_rs = [allXtoR[x] for x in offline_xs]


    # Not subgfn: sample trajectories from backward policy
    with torch.no_grad():
      offline_trajs = self.model.batch_back_sample(offline_xs)

    
    offline_trajs = [
      Experience(traj=traj, x=x, r=r,
                logr=torch.log(torch.tensor(r, dtype=torch.float32,device=self.args.device))
                )
      for traj, x, r in zip(offline_trajs, offline_xs, offline_rs)
    ]
    
    return offline_trajs

 
