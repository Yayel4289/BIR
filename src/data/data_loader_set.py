from typing import List
from torch.utils.data import DataLoader

from exps.points_classification.utils.point import Point


class DataLoaderSet():
    def __init__(
            self,
            source_train_loader: DataLoader, source_val_loader: DataLoader, source_test_loader: DataLoader,
            target_train_loader: DataLoader, target_val_loader: DataLoader, target_test_loader: DataLoader,
            test_pts: List[Point] = None
        ) -> None:
        self.source_train_loader = source_train_loader
        self.source_val_loader = source_val_loader
        self.source_test_loader = source_test_loader
        self.target_train_loader = target_train_loader
        self.target_val_loader = target_val_loader
        self.target_test_loader = target_test_loader
        self.test_pts = test_pts

    def __iter__(self):
        return iter((
            self.source_train_loader, self.source_val_loader, self.source_test_loader, 
            self.target_train_loader, self.target_val_loader, self.target_test_loader, 
        ))

    def has_test_pts(self):
        return self.test_pts is not None

    def get_test_pts(self):
        return self.test_pts
