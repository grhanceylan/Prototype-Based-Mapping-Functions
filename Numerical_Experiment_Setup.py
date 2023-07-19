from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np

# this functions generates random states for nested search
def gen_random_states(random_state=42, number_of_executions=10):
    rng = np.random.RandomState(seed=random_state)
    seeds = np.arange(10 ** 5)
    rng.shuffle(seeds)
    return seeds[:number_of_executions]



def run(data,pipe_line, grid_search_params):
    performance_metrics = { 'Fold_Num':[], 'Train_Acc': [], 'Test_Acc': [], 'Train_Time': [],'Train_F1':[],'Test_F1':[], 'N_Proto':[]}
    meta_models = {'Fold_Num': [], 'Model': []}
    X, y = data()
    # if true then training and test parts are provided in repository
    if len(X)==2:
        random_seeeds = [1]
        X_train, X_test, y_train, y_test= X[0], X[1], y[0], y[1]

    else:
        random_seeeds = gen_random_states(random_state=42, number_of_executions=10)

    foldnum = 0
    for rs in random_seeeds:
        print('Fold Num: ', foldnum+1)
        if len(X) != 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                                        shuffle=True, random_state=rs,
                                                                        stratify=y)

        grid_search = GridSearchCV(estimator=pipe_line, param_grid=grid_search_params, refit=True, cv=2 , scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        performance_metrics['Train_Time'].append(grid_search.refit_time_)
        preds = grid_search.best_estimator_.predict(X_train)
        performance_metrics['Train_Acc'].append(accuracy_score(y_train, preds))
        performance_metrics['Train_F1'].append(
            f1_score(y_train,preds, average='weighted' if len(np.unique(y_train))>2 else 'binary'))
        preds = grid_search.best_estimator_.predict(X_test)
        performance_metrics['Test_Acc'].append(accuracy_score(y_test,preds))
        performance_metrics['Test_F1'].append(
            f1_score(y_test, preds, average='weighted' if len(np.unique(y_train)) > 2 else 'binary'))

        if 'mapping' in pipe_line.get_params():
            if grid_search.best_estimator_['mapping'].get_params()['feature_map'].mapping=='Phi_M' or grid_search.best_estimator_['mapping'].get_params()['feature_map'].mapping=='Phi_MN':
                t=0
                print(grid_search.best_estimator_['mapping'].get_params()['feature_map'].mapping)
                for i in range(len(np.unique(y_train))):
                    t+=len(grid_search.best_estimator_['mapping'].proto[i])
                performance_metrics['N_Proto'].append(t)
            else:
                print(grid_search.best_estimator_['mapping'].get_params()['feature_map'].mapping)
                performance_metrics['N_Proto'].append(
                    len(grid_search.best_estimator_['mapping'].proto))
        else:
            performance_metrics['N_Proto'].append(0)
        performance_metrics['Fold_Num'].append(foldnum)
        meta_models['Fold_Num'].append(foldnum)
        meta_models['Model'].append(grid_search.best_estimator_)
        foldnum = foldnum + 1
    return performance_metrics, meta_models



