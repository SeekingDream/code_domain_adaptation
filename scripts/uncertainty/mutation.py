from scripts.uncertainty import BasicUncertainty
from scripts.utils import common_predict
import numpy as np


class Mutation(BasicUncertainty):
    def __init__(self, model, train_loader, test_loader, train_ground, test_ground, device):
        super(Mutation, self).__init__(model, train_loader, test_loader, train_ground, test_ground, device)

    def get_uncertainty(self, iter_time=5):
        train_score, test_score = 0, 0

        _, train_predict, _ = \
            common_predict(self.model, self.train_loader, self.device)
        _, test_predict, _ = \
            common_predict(self.model, self.test_loader, self.device)

        for i in range(iter_time):
            self.model.train()
            _, train_label, _ = \
                common_predict(self.model, self.train_loader, self.device)
            train_score += (train_label == train_predict)
        train_score = train_score / iter_time

        for i in range(iter_time):
            self.model.train()
            _, test_label, _ = \
                common_predict(self.model, self.test_loader, self.device)
            test_score += (test_label == test_predict)
        test_score = test_score / iter_time

        self.model.eval()
        return train_score, test_score
