import pickle
import numpy as np


if __name__ == "__main__":
    for length in [14, 50, 100]:
        for task in range(1, 5):
            with open(f"datasets/L{length}_RNA{task}/rewards.pkl", "rb") as f:
                reward = pickle.load(f)
            print(len(reward))
            print(f"L{length}_RNA{task}: {np.percentile(reward, 100*(1-0.001)):.3f}")
