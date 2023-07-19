import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

# this class serves as a wrapper function that returns a scikit-learn transformer from given prototype based feature map
class ProtoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_map=None, proto=None, n_proto=None, **kwargs):
        # given feature map
        self.feature_map = feature_map
        # predetermined anchors  (currently it is not utilized)
        self.proto = proto
        # number of anchor sets
        self.n_proto = n_proto

    def fit(self, X, y=None):
        # assign anchor selection method
        self.proto_selection = KMeans(init='k-means++', n_clusters=self.n_proto, random_state=42)

        if self.feature_map.multi_proto_set:
            self.proto = []

            quantiles = []

            for q in  [0.25, 0.5, 0.75, 1]:
                quantiles.append(np.quantile(y,q))
            self.proto.append(self.proto_selection.fit(X[y<=quantiles[0]]).cluster_centers_)
            for qi in range(1,len(quantiles)-1):
                inds_1= y>quantiles[qi]
                inds_2= y<=quantiles[qi+1]
                inds = [inds_1[i]*inds_2[i] for i in range(len(inds_1))]
                self.proto.append(self.proto_selection.fit(X[inds]).cluster_centers_)

        else:
            self.proto = self.proto_selection.fit(X).cluster_centers_
        return self

    def transform(self, X):
        return self.feature_map.map(X, self.proto)

