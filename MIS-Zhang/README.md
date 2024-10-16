# Note

The code is based on [https://github.com/zdhNarsil/GFlowNet-CombOpt](https://github.com/zdhNarsil/GFlowNet-CombOpt)

### Requirements

```bash
pip install hydra-core==1.1.0 omegaconf submitit hydra-submitit-launcher
pip install dgl==0.6.1
```

### How to run?

You can fix the setting at 'configs/main.yaml'

The code for pessimistic training of backward policy is implemented in 'code/algorithm.py'

```python
if self.cfg.alg == 'pbp':
    self.back_opt.zero_grad()
    back_logits = self.back_model(gb_two, s_two, reward_exp)
    pb_logits = back_logits[total_num_nodes:, ..., 0]
    pb_logits[~get_decided(s_next)] = -np.inf
    pb_logits = pad_batch(pb_logits, numnode_per_graph, padding_value=-np.inf)
    log_pb_upt = F.log_softmax(pb_logits, dim=1)[torch.arange(batch_size), a]
    log_pb = log_pb_upt.clone().detach()
    torch.mean(-log_pb_upt).backward()
    self.back_opt.step()
```


You can generate data by

```bash
cd data/
python rbgraph_generator.py --num_graph 4000 --save_dir rb200-300/train
python rbgraph_generator.py --num_graph 500 --save_dir rb200-300/test
```

You can run your experiment by 

```bash
cd code/
python main.py --config-name main
```