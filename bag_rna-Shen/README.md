# Note
Our implementation is based on "Towards Understanding and Improving GFlowNet Training" (https://github.com/maxwshen/gflownet). 


### How to run?

You can fix the setting at 'exps/[task]/setting.yaml'

The code for pessimistic training of backward policy is implemented in 'gflownet/trainer.py'

```python
# Pessimistic training of backward policy
for step_num in range(self.args.alg_N):
    if self.args.model in ['pbp']:
        self.model.train_pbp(buffer_for_PBP[-random.randint(0,len(buffer_for_PBP)-1)])
```

You can run your experiment by 

```
python runexpwb.py --setting tfbind8 --model pbp 
python runexpwb.py --setting rna --model pbp --rna_task 1 
python runexpwb.py --setting rna --model pbp --rna_task 2 
python runexpwb.py --setting rna --model pbp --rna_task 3 
```
