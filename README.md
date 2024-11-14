## *The overall code is currently being organized.*


# Repository Overview

We provide the PyTorch implementation for **PBP-GFN** (NeurIPS 2024, [https://arxiv.org/abs/2405.16012](https://arxiv.org/abs/2405.16012)) framework here. Our implementations are based on the settings of various prior studies, including both on-policy and off-policy training, as described in **Appendix B** of our paper. 
 
```
|-- {MIS-Zhang, Molecule-Recursion, bag_rna-Shen} # PBP-GFN for Maximum independent set, Molecular generation, and (RNA-sequence, bag) generation
|-- grid_PBP-GFN.py # PBP-GFN for 16x16x16 hyper grid
```

---

## Maximum independent set, Molecule, Bag, and RNA sequence

Check the detailed implementations for these tasks in each directory.

---


## Hyper-grid 

The code for hyper-grid environment follows [https://gist.github.com/malkin1729/9a87ce4f19acdc2c24225782a8b81c15](https://gist.github.com/malkin1729/9a87ce4f19acdc2c24225782a8b81c15) implemented by [Malkin et al.](https://arxiv.org/abs/2201.13259)

You can run your experiment by 

```bash
python grid_PBP-GFN.py
```
