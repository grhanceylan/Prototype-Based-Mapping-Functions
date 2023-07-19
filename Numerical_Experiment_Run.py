import os
import numpy as np
# sklearn classifiers
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RidgeClassifier as Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
# custom classes
import DataReader
import Helpers
from PrototypeBasedFeaturemaps import Featuremap
from Numerical_Prototype_Selection import ProtoTransformer
import Numerical_Experiment_Setup as Experiment

# create pipeline
def get_pipeline(scaler=None, feature_map=None, clf=None):
        if feature_map == 'Linear':
            return Pipeline(steps=[('scale', scaler), ('clf', clf)])
        return Pipeline(steps=[('scale', scaler),
                               ('mapping', ProtoTransformer(feature_map=feature_map)),
                               ('clf', clf)])
# datasets
datasets=[ DataReader.australian, DataReader.ionosphere, DataReader.heart, DataReader.bupa, DataReader.wdbc,
           DataReader.spambase,DataReader.phoneme,DataReader.letter, DataReader.abalone, DataReader.magic,
           DataReader.mammography, DataReader.pendigits, DataReader.shuttle,DataReader.optdigits,DataReader.usps,
           DataReader.splice,DataReader.wilt,DataReader.fourclass,DataReader.monk3,DataReader.phishing]


base_classifiers=[LDA(),
                  LinearSVC(dual=False,random_state=42),
                  LR(random_state=42,solver='liblinear', dual=False),
                  Ridge(random_state=42)]

# scaling function 
experiment_scaler = MinMaxScaler()
# this to be used on mapping functions
metric='l1'
feature_maps=['Linear',
              Featuremap(mapping='Phi_1', metric=metric), Featuremap(mapping='Phi_M', metric=metric),
              Featuremap(mapping='Phi_N', metric=metric),Featuremap(mapping='Phi_MN', metric=metric) ]

# number of prototypes for each prototype sets
n_proto=[1,2,3]

# regularization parameters for the base classifiers
c_params= np.power(10.0, range(-6, 6))

# grid search params
grid_params=[ [{},
               {'mapping__n_proto':n_proto},{'mapping__n_proto':n_proto},
               {'mapping__n_proto':n_proto},{'mapping__n_proto':n_proto}],
             
              [{'clf__C':c_params},
               {'clf__C':c_params,'mapping__n_proto':n_proto},{'clf__C':c_params,'mapping__n_proto':n_proto},
               {'clf__C':c_params,'mapping__n_proto':n_proto},{'clf__C':c_params,'mapping__n_proto':n_proto}],
             
              [{'clf__C': c_params},
               {'clf__C': c_params, 'mapping__n_proto': n_proto}, {'clf__C': c_params, 'mapping__n_proto': n_proto},
               {'clf__C': c_params, 'mapping__n_proto': n_proto}, {'clf__C': c_params, 'mapping__n_proto': n_proto}],
             
              [{'clf__alpha': c_params},
               {'clf__alpha': c_params, 'mapping__n_proto': n_proto}, {'clf__alpha': c_params, 'mapping__n_proto': n_proto},
               {'clf__alpha': c_params, 'mapping__n_proto': n_proto}, {'clf__alpha': c_params, 'mapping__n_proto': n_proto}],
             ]

# create a directory to write performance results
directory = Helpers.generate_directory_name()
os.makedirs(directory)


read_me_text = []
if experiment_scaler is not None:
    read_me_text.append("------------- Scale:\t" + str(experiment_scaler) + "\t" + str(experiment_scaler.get_params()))
else:
      read_me_text.append("------------- Scale:\t" + str(experiment_scaler))
read_me_text.append("------------- Metric:\t" + metric)

for b, base_clf in enumerate(base_classifiers):
    model_name = str(base_clf).split('(')[0]
    print(model_name)
    for f, f_map in enumerate(feature_maps):
        mapping = f_map.mapping if f_map != 'Linear' else 'Linear'
        print("------------- Model: ", model_name, " Feature Map:\t", mapping+ "\t"+str(grid_params[b][f] ))
        read_me_text.append(
            str("------------- Model:\t" + model_name + "\tFeature Map:\t" + mapping + "\t" + str(grid_params[b][f])))
        for d, data in enumerate(datasets):
            data_name=data.__name__.upper()
            print(data_name)
            pipe_line=get_pipeline(scaler=experiment_scaler,feature_map=f_map,clf=base_clf)
            performance_metrics,_ = Experiment.run(data=data, pipe_line=pipe_line, grid_search_params=grid_params[b][f])
            print(performance_metrics)
            Helpers.save_performance_metrics(dataset=data_name, model=model_name, feature_map=mapping, performance_metrics = performance_metrics, directory=directory)
        


# save read_me text
print(Helpers.save_read_me_text(read_me_text,directory))
# print obtained results in a latex table format
Helpers.result_summarizer(path= directory+'/Results.csv')
