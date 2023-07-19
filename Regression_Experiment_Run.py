import os
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# custom classes
import DataReader
import Helpers
from PrototypeBasedFeaturemaps import Featuremap
from Regression_Prototype_Selection import ProtoTransformer
import Regression_Experiment_Setup as Experiment


def get_pipeline(scaler=None,feature_map=None,clf=None):
    if feature_map == 'Linear':
        if scaler is not None:
            return Pipeline(steps=[('scale', scaler), ('clf', clf)])
        else:
            return Pipeline(steps=[('clf', clf)])
    else:
        if scaler is not None:
            return Pipeline(steps=[('scale', scaler),
                                   ('mapping', ProtoTransformer(feature_map=feature_map)),
                                   ('clf', clf)])
        else:
            return Pipeline(steps=[('mapping', ProtoTransformer(feature_map=feature_map)),
                                   ('clf', clf)])

experiment_scaler = None

datasets=[ DataReader.R_abalone, DataReader.R_airfoil, DataReader.R_ccp,DataReader.R_housing,DataReader.R_superconduct,DataReader.R_energy]
datasets=[ DataReader.R_abalone]
base_classifiers=[LinearRegression()]
metric='l1'
feature_maps=['Linear',
              Featuremap(mapping='Phi_1', metric=metric), Featuremap(mapping='Phi_M', metric=metric),
              Featuremap(mapping='Phi_N', metric=metric),Featuremap(mapping='Phi_MN', metric=metric) ]


n_proto=[1,2,3,4,5]

grid_params=[
              [{},
               {'mapping__n_proto':n_proto},{'mapping__n_proto':n_proto},   {'mapping__n_proto':n_proto},{'mapping__n_proto':n_proto},
              ],

        ]

# create a directory to write performance results
directory = Helpers.generate_directory_name()
os.makedirs(directory)

for b, base_clf in enumerate(base_classifiers):
    model_name=str(base_clf).split('(')[0]
    for f,f_map in enumerate(feature_maps):
        mapping= f_map.mapping if f_map != 'Linear' else 'Linear'
        print("****************** Model: ", model_name, " Feature Map:\t", mapping)
        for data in datasets:
            data_name=data.__name__.upper()
            print(data_name)
            pipe_line=get_pipeline(scaler=experiment_scaler,feature_map=f_map,clf=base_clf)
            performance_metrics = Experiment.run(data=data, pipe_line=pipe_line, grid_search_params=grid_params[b][f])
            Helpers.save_performance_metrics(dataset=data_name, model=model_name, feature_map=mapping,
                                             performance_metrics=performance_metrics, directory=directory)





Helpers.result_summarizer(path= directory+'/Results.csv')

