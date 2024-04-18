import torch

############
# datasets #
############
class ToyDataset:
    def __init__(self, n_samples=256, tokens=255, dtype=torch.long):
        from sklearn.datasets import make_moons
        x, _ = make_moons(n_samples=n_samples, shuffle=True, noise=None, random_state=None)
        x = x - x.min()
        x = x / x.max()
        x = (tokens * x).astype(int)
        assert x.max() <= tokens and x.min() >= 0
        self.x = torch.from_numpy(x).to(dtype)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.size(0)