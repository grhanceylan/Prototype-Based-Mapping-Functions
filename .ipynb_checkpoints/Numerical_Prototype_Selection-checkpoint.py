import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels


# this class serves as a wrapper function that returns a scikit-learn transformer from given prototype based feature map
class ProtoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_map=None, n_proto=None, **kwargs):
        # given feature map
        self.feature_map = feature_map
        # number prototypes for each prototype sets
        self.n_proto= n_proto

    def fit(self, X, y=None):
        self.classes_ = unique_labels(y)
        #self.proto_selection = KMeans(init='k-means++', n_clusters=self.n_proto, random_state=42)

        #self.proto_selection=  MDD_Critic(number_of_protos=self.n_proto)
        self.proto_selection = KMedoids(init='k-medoids++', n_clusters=self.n_proto, random_state=42)
        # apply k-means and set centers as prototypes
        if self.feature_map.multi_proto_set:
            self.proto = []
            # apply k-means for each class
            for yi in self.classes_:

                    self.proto.append(self.proto_selection.fit(X[y == yi]).cluster_centers_)
        else:
            # apply k-means

            self.proto=self.proto_selection.fit(X).cluster_centers_

        return self

    def transform(self, X):
        return self.feature_map.map(X, self.proto)

