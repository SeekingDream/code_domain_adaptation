from scripts.uncertainty import BasicUncertainty
from scripts.utils import common_predict
import numpy as np


class Vinalla(BasicUncertainty):
    def __init__(self, model, train_loader, test_loader, train_ground, test_ground, device):
        super(Vinalla, self).__init__(model, train_loader, test_loader, train_ground, test_ground, device)

    def get_uncertainty(self):
        train_score, _, _ = \
            common_predict(self.model, self.train_loader, self.device)
        #train_score = np.max(train_score, axis=1)
        test_score, _, _ = \
            common_predict(self.model, self.test_loader, self.device)
        #test_score = np.max(test_score, axis=1)
        return train_score, test_score
