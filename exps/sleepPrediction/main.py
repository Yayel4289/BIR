from pyhealth.datasets import SHHSDataset

dataset = SHHSDataset(root="/srv/local/data/SHHS/")
print(dataset.stat())
dataset.info()

