from torch.utils.data import Sampler, TensorDataset

class CyclingSampler(Sampler):
    def __init__(self, data_source: TensorDataset, batch_size: int):
        self.data_source = data_source
        self.batch_size = batch_size
        self.current_idx = 0
    
    def __iter__(self):
        indices = []
        for _ in range(self.batch_size):
            indices.append(self.current_idx)
            self.current_idx = (self.current_idx + 1) % len(self.data_source)
        return iter(indices)
    
    def __len__(self):
        return len(self.data_source)
