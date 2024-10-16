# Repository Overview
We provide the PyTorch implementation for PBP-GFN framework here. Our implementations are based on the prior studies. The overall code is currently being organized.

```python
|-- {MIS-Zhang, Molecule-Recursion, bag_rna-Shen} # PBP-GFN for Maximum independent set, Molecular generation, and RNA-sequence generation
|-- grid_PBP-GFN.py # PBP-GFN for 16x16x16 hyper grid
```

---

## Hyper-grid envrionemnt

The implementations in hyper-grid environment are based on [https://gist.github.com/malkin1729/9a87ce4f19acdc2c24225782a8b81c15](https://gist.github.com/malkin1729/9a87ce4f19acdc2c24225782a8b81c15) released by [Malkin et al.](https://arxiv.org/abs/2201.13259)

You can run your experiment by 

```bash
python grid_PBP-GFN.py
```

---

## Others

One can check the detailed implementations for other tasks in each directory.
