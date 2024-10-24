# Note

The code is based on [https://github.com/recursionpharma/gflownet](https://github.com/recursionpharma/gflownet)


### How to run?


The code for pessimistic training of backward policy is implemented in 'trainer.py' and 'algo/pbp_tb.py'

```python
# Pessimistic training of GFlowNets
if self.cfg.algo.method == "PBP_TB": 
    fixed_size_buffer.append((batch.to(self.device),None))
    fixed_size_buffer = fixed_size_buffer[-self.cfg.algo.buffer_size:]     
    for fpb_step in range(self.cfg.algo.alg_N):
        self.model.train()
        batch_sampled, traj_log_p_sampled = fixed_size_buffer[random.randint(0,len(fixed_size_buffer)-1)]
        loss = self.algo.compute_batch_losses_for_FPB(
            self.model, batch_sampled, traj_log_p_sampled)
        step_info = self.step_pbp(loss)
```


You can run your experiment by 

```bash
cd tasks/
python python seh_frag_our.py
```