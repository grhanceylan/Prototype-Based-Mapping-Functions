from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
numOutCV: int = 5
numInCV: int = 4
random_state: int=16252329

def run(data,pipe_line, grid_search_params):
    performance_metrics = {'Fold_Num': [],'Train_Time':[] ,'Test_MSE': [], 'N_Proto': []}
    X, y = data()

    y = StandardScaler().fit_transform(np.expand_dims(y, -1))
    y = np.reshape(y, (len(y),))

    skf = KFold(n_splits=numOutCV, shuffle=True, random_state=random_state)
    foldnum = 0

    for train_index, test_index in skf.split(X, y):
        print('Fold Num: ', foldnum + 1)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        SC = StandardScaler()
        X_train = SC.fit_transform(X_train)
        X_test = SC.transform(X_test)

        inner_cv = KFold(n_splits=numInCV, shuffle=True, random_state=random_state)
        gcv = GridSearchCV(estimator=pipe_line, param_grid=grid_search_params,
                           n_jobs=-1, cv=inner_cv, verbose=0, refit=True)
        grid_search = gcv.fit(X_train, y_train)
        gcv_pred = grid_search.best_estimator_.predict(X_test)


        performance_metrics['Test_MSE'].append(mean_squared_error(gcv_pred, y_test))
        performance_metrics['Train_Time'].append(grid_search.refit_time_)
        if 'mapping' in pipe_line.get_params():
            if grid_search.best_estimator_['mapping'].get_params()['feature_map'].mapping == 'Phi_M' or \
                    grid_search.best_estimator_['mapping'].get_params()['feature_map'].mapping == 'Phi_MN':

                performance_metrics['N_Proto'].append( 4*len(grid_search.best_estimator_['mapping'].proto[0]))
            else:
                print(grid_search.best_estimator_['mapping'].get_params()['feature_map'].mapping)
                performance_metrics['N_Proto'].append(
                    len(grid_search.best_estimator_['mapping'].proto))
        else:
            performance_metrics['N_Proto'].append(0)
        performance_metrics['Fold_Num'].append(foldnum)

        foldnum +=1

    return performance_metrics


