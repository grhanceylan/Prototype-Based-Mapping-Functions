import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min as dist
from sklearn.metrics import pairwise_distances
class Featuremap:
    def __init__(self, mapping='Phi_1', metric='l1'):
        # mapping function name should be one of the followings: 'Phi_1', 'Phi_M', 'Phi_N', 'Phi_MN
        self.mapping = mapping
        #  metric function to calculate distance
        self.metric = metric
        # is multi prototype sets supported or not
        self.multi_proto_set = False
        # set mapping function and multi prototype properties
        self.mapping_func = self._set_params()

    def _set_params(self):
        if self.mapping == 'Phi_1':
            self.multi_proto_set = False
            return self.phi_1
        elif self.mapping == 'Phi_M':
            self.multi_proto_set = True
            return self.phi_M
        elif self.mapping == 'Phi_N':
            self.multi_proto_set = False
            return self.phi_N
        elif self.mapping == 'Phi_MN':
            self.multi_proto_set = True
            return self.phi_MN
        elif self.mapping == 'pekalska':
            self.multi_proto_set= False
            return self.pekalska_representation
    def map(self, X, protos):
        return self.mapping_func(X, protos)

    def phi_1(self, X, protos):
        return np.hstack((X,dist(X, protos, metric=self.metric)[1].reshape((len(X), 1))))

    def phi_M(self, X, proto_sets):
        d = np.zeros((len(X), len(proto_sets)))
        i = 0
        for protos in proto_sets:
            d[:, i] = dist(X, protos, metric=self.metric)[1]
            i = i + 1
        return np.hstack((X, d))

    def phi_N(self, X, protos):
        inds = dist(X, protos, metric=self.metric)[0]
        if self.metric == 'l1':
            return np.hstack((X, np.abs(X - protos[inds])))
        elif self.metric == 'l2':
            return np.hstack((X, np.abs(X - protos[inds]) ** 2))

    def phi_MN(self, X, proto_sets):
        d = np.zeros((len(X), len(proto_sets) * len(X[0])))
        i = 0
        for protos in proto_sets:
            temp = dist(X, protos, metric=self.metric)
            d[:, i * len(X[0]):(i + 1) * len(X[0])] = np.abs(X - protos[temp[0]]) if self.metric== 'l1' else  np.abs(X - protos[temp[0]])**2
            i = i + 1
        return np.hstack((X, d))

    def pekalska_representation(self,X, protos):
        return   pairwise_distances(X,protos, metric=self.metric)