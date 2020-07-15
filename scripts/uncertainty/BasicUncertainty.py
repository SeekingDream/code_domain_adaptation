from abc import ABCMeta, abstractmethod
import numpy as np


class BasicUncertainty:
    def __init__(
            self, model, train_loader, test_loader,
            train_ground, test_ground, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_ground = train_ground
        self.test_ground = test_ground
        self.device = device
        self.train_score, self.test_score = self.get_uncertainty()



    @abstractmethod
    def get_uncertainty(self):
        return None, None

    def get_thresh(self, requirement):
        train_score = np.sort(self.train_score)
        for score in train_score:
            select = (self.train_score > score)
            acc = np.sum(select * self.train_ground) / (np.sum(select) + 1e-5)
            if acc > requirement:
                return acc, score
        return acc, score

    def get_coverindex(self, requirement):
        acc, score = self.get_thresh(requirement)
        select = (self.test_score > score)
        acc = np.sum(select * self.test_ground) / (np.sum(select) + 1e-5)
        return acc, np.where(select == 1)



