import numpy as np
import pandas as pd

from sklearn.datasets import load_svmlight_file

def heloc():
    df = pd.read_csv(filepath_or_buffer='Datasets/heloc_dataset_v1.csv', delimiter=',')

    df.drop_duplicates(inplace=True)

    y=df['RiskPerformance']
    df.drop(columns=['RiskPerformance'],inplace=True, axis=1)
    y = y[(df >= 0).all(axis=1)]
    df = df.loc[(df >= 0).all(axis=1)]
    y[y=='Bad']=1
    y[y == 'Good'] = 0

    return df.values, np.array(y.values, dtype=int)

def german():
    df = pd.read_csv(filepath_or_buffer='Datasets/german_numeric.txt', header= None, delim_whitespace=True)
    df = np.array(df)
    df[:, -1][df[:, -1]==2]=0

    return df[:, 0:-1], df[:, -1]
X,y=german()



def boston():
    # http://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
    df = pd.read_csv(filepath_or_buffer='Datasets/boston_housing.txt', delim_whitespace=True)
    df = np.array(df)
    temp_1 = df[:, -1] <= np.median(df[:, -1])
    temp_2 = df[:, -1] > np.median(df[:, -1])
    df[:, -1][temp_1] = 0
    df[:, -1][temp_2] = 1
    return df[:, 0:-1], df[:,-1]

def a9a():
    X = load_svmlight_file('Datasets/a9a_train.txt', n_features=123)
    X_train, y_train = np.asarray(X[0].todense()), np.array(X[1], dtype=int)

    X = load_svmlight_file('Datasets/a9a_test.txt', n_features=123)
    X_test, y_test = np.asarray(X[0].todense()), np.array(X[1], dtype=int)

    return [X_train, X_test], [y_train, y_test]

def adult():
    # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    X = load_svmlight_file('Datasets/adult_train_libsvm.txt',n_features=123)
    X_train, y_train = np.asarray(X[0].todense()), np.array(X[1], dtype=int)

    X = load_svmlight_file('Datasets/adult_test_libsvm.txt',n_features=123)
    X_test, y_test = np.asarray(X[0].todense()), np.array(X[1], dtype=int)

    return [X_train, X_test], [y_train, y_test]

# -------------------------------------------------------------------------------------
def california_housing():
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing
    df = pd.read_csv(filepath_or_buffer='Datasets/california_housing.csv', delimiter=',')
    df = np.array(df)


    temp_1= df[:, -1]<np.median(df[:, -1])
    temp_2= df[:, -1]>=np.median(df[:, -1])

    df[:, -1][temp_1]=1
    df[:, -1][temp_2]=0

    return df[:, 1:-1], df[:,-1]

# -------------------------------------------------------------------------------------
def german_credit():
    # http://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
    df = pd.read_csv(filepath_or_buffer='Datasets/german_credit_numeric.txt', delimiter=',')

    df = np.array(df)
    return df[:, 1:], df[:,0]

# -------------------------------------------------------------------------------------
def phishing():
    # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    X = load_svmlight_file('Datasets/phishing.txt')
    return np.asarray(X[0].todense()), np.array(X[1], dtype=int)
# -------------------------------------------------------------------------------------
def monk3():
    # https://archive.ics.uci.edu/ml/datasets/MONK's+Problems
    df = pd.read_csv(filepath_or_buffer='Datasets/monks3_train.txt', header=None, delimiter=' ')
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns) - 1)]
    df.drop(columns=df.columns[-1],axis=1, inplace=True)
    df =np.asarray(df)
    X_train,y_train=df[:,2:],df[:,1]

    df = pd.read_csv(filepath_or_buffer='Datasets/monks3_test.txt', header=None, delimiter=' ')
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns) - 1)]
    df.drop(columns=df.columns[-1], axis=1, inplace=True)
    df = np.asarray(df)
    X_test, y_test = df[:, 2:], df[:, 1]
    return [X_train, X_test], [y_train, y_test]
# -------------------------------------------------------------------------------------
def fourclass():
    # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    X = load_svmlight_file('Datasets/fourclass.txt')
    return np.asarray(X[0].todense()), np.array(X[1], dtype=int)
# -------------------------------------------------------------------------------------
def splice():
    # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    X = load_svmlight_file('Datasets/splice_train.txt')
    X_train, y_train = np.asarray(X[0].todense()), np.array(X[1], dtype=int)

    X = load_svmlight_file('Datasets/splice_test.txt')
    X_test, y_test = np.asarray(X[0].todense()), np.array(X[1], dtype=int)
    return [X_train, X_test], [y_train, y_test]

# -------------------------------------------------------------------------------------
def wilt():
    # https://archive.ics.uci.edu/ml/datasets/wilt
    df = pd.read_csv(filepath_or_buffer='Datasets/wilt_training.csv', delimiter=',', header=None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns) - 1)]

    y = pd.factorize(df['y'])[0]
    df.drop('y', axis=1, inplace=True)
    df['y'] = y
    df = np.array(df)
    X_train, y_train= df[:, 0:-1], df[:, -1]
    df = pd.read_csv(filepath_or_buffer='Datasets/wilt_testing.csv', delimiter=',', header=None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns) - 1)]

    y = pd.factorize(df['y'])[0]

    df.drop('y', axis=1, inplace=True)
    df['y'] = y
    df = np.array(df)
    X_test, y_test = df[:, 0:-1], df[:, -1]
    return [X_train,X_test],[y_train,y_test]
# -------------------------------------------------------------------------------------
def usps():
    # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
    X = load_svmlight_file('Datasets/usps_train')
    X_train, y_train= np.asarray(X[0].todense()),np.array(X[1], dtype=int)
    X = load_svmlight_file('Datasets/usps_test')
    X_test, y_test = np.asarray(X[0].todense()), np.array(X[1], dtype=int)
    return [X_train,X_test],[y_train,y_test]

# -------------------------------------------------------------------------------------
def optdigits():
    # http://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
    df = pd.read_csv(filepath_or_buffer='Datasets/optdigits_train.csv', delimiter=',', header=None)
    df = np.array(df)
    X_train, y_train= df[:, 0:-1], df[:, -1]
    df = pd.read_csv(filepath_or_buffer='Datasets/optdigits_test.csv', delimiter=',', header=None)
    df = np.array(df)
    X_test, y_test = df[:, 0:-1], df[:, -1]
    return [X_train,X_test],[y_train,y_test]
# -------------------------------------------------------------------------------------
def australian():
    # https://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)
    df = pd.read_csv(filepath_or_buffer='Datasets/australian.csv', delimiter=',', header=None)
    df = np.array(df)
    return df[:, 0:-1], df[:, -1]
# -------------------------------------------------------------------------------------
def ionosphere():
    # https://archive.ics.uci.edu/ml/datasets/ionosphere
    df = pd.read_csv(filepath_or_buffer='Datasets/ionosphere.csv', delimiter=',', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df['y']= pd.factorize(df['y'])[0]
    df = np.array(df)
    return df[:, :-1], df[:, -1]
# -------------------------------------------------------------------------------------
def heart():
    # https://archive.ics.uci.edu/ml/datasets/statlog+(heart)
    df = pd.read_csv(filepath_or_buffer='Datasets/heart.csv', delimiter=',', header=None)
    df = np.array(df)
    return df[:, 0:-1], df[:, -1]
# -------------------------------------------------------------------------------------
def bupa():
    # https://archive.ics.uci.edu/ml/datasets/liver+disorders
    df = pd.read_csv(filepath_or_buffer='Datasets/bupa.csv', delimiter=',', header=None)
    df = np.array(df)
    # duplicated rows: 84-86, 141-318, 143-150, 170-176
    # remove duplicates
    df = np.delete(df, [84, 141, 143, 170], axis=0)
    return df[:, 0:-1], df[:, -1]

# -------------------------------------------------------------------------------------
def spambase():
    # https://archive.ics.uci.edu/ml/datasets/spambase
    df = pd.read_csv(filepath_or_buffer='Datasets/spambase.csv', delimiter=',', header=None)
    df = np.array(df)
    return df[:, 0:-1], df[:, -1]
# -------------------------------------------------------------------------------------
def phoneme():
    # https://machinelearningmastery.com/predictive-model-for-the-phoneme-imbalanced-classification-dataset/
    df = pd.read_csv(filepath_or_buffer='Datasets/phoneme.csv', delimiter=',', header=None)
    df = np.array(df)
    return df[:, 0:-1], df[:, -1]
# -------------------------------------------------------------------------------------
def letter():
    # https://archive.ics.uci.edu/ml/datasets/letter+recognition
    # multiclass
    df = pd.read_csv(filepath_or_buffer='Datasets/letter_recognition.csv', delimiter=',', header=None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(1, len(df.columns))]
    y = pd.factorize(df['y'])[0]
    df.drop('y', axis=1, inplace=True)
    df['y'] = y
    df = np.array(df)
    return df[:, 0:-1], df[:, -1]
# -------------------------------------------------------------------------------------
def abalone():
    # https://archive.ics.uci.edu/ml/datasets/abalone
    # multiclass
    df = pd.read_csv(filepath_or_buffer='Datasets/abalone.csv', delimiter=',', header=None)
    df.columns = ['y'] + ['X_' + str(i) for i in range(1, len(df.columns))]
    y = pd.factorize(df['y'])[0]
    df.drop('y', axis=1, inplace=True)
    df['y'] = y
    df = np.array(df)
    return df[:, 0:-1], df[:, -1]

# -------------------------------------------------------------------------------------
def magic():
    # https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope
    df = pd.read_csv(filepath_or_buffer='Datasets/magic.csv', delimiter=',', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df['y'] = pd.factorize(df['y'])[0]
    df = np.array(df)
    return df[:, 0:-1], df[:, -1]
# -------------------------------------------------------------------------------------
def mammography():
    # https://www.openml.org/d/310
    df = pd.read_csv(filepath_or_buffer='Datasets/mammography.csv', delimiter=',', header=None)
    df = np.array(df)
    return df[:, 0:-1], df[:, -1]
# -------------------------------------------------------------------------------------
def banknote():
    # https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    df = pd.read_csv(filepath_or_buffer='Datasets/banknote.csv', delimiter=',', header=None)
    df = np.array(df)
    return df[:, 0:-1], df[:, -1]
# -------------------------------------------------------------------------------------
def wdbc():
    # https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    df = pd.read_csv(filepath_or_buffer='Datasets/wdbc.csv', delimiter=',', header=None, index_col=0)
    df.columns = ['y'] + ['X_' + str(i) for i in range(len(df.columns) - 1)]
    y = pd.factorize(df['y'])[0]
    df.drop('y', axis=1, inplace=True)
    df['y'] = y
    df = np.array(df)
    return df[:, 0:-1], df[:, -1]


# -------------------------------------------------------------------------------------
def shuttle():
    # https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)
    df = pd.read_csv(filepath_or_buffer='Datasets/shuttle_train.csv', delimiter=',', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df = np.array(df)
    X_train, y_train= df[:, 0:-1], df[:, -1]
    df = pd.read_csv(filepath_or_buffer='Datasets/shuttle_test.csv', delimiter=',', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df = np.array(df)
    X_test, y_test = df[:, 0:-1], df[:, -1]
    return [X_train,X_test],[y_train,y_test]
# -------------------------------------------------------------------------------------
def pendigits():
    # http://archive.ics.uci.edu/ml/datasets/pen-based+recognition+of+handwritten+digits
    df = pd.read_csv(filepath_or_buffer='Datasets/pendigits_train.csv', delimiter=',', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df = np.array(df)
    X_train, y_train= df[:, 0:-1], df[:, -1]
    df = pd.read_csv(filepath_or_buffer='Datasets/pendigits_test.csv', delimiter=',', header=None)
    df.columns = ['X_' + str(i) for i in range(len(df.columns) - 1)] + ['y']
    df = np.array(df)
    X_test, y_test = df[:, 0:-1], df[:, -1]
    return [X_train,X_test],[y_train,y_test]
# regression_datasets

def R_abalone():
    df = pd.read_csv(filepath_or_buffer='Datasets/Regression/abalone.csv', delimiter=',', header=None)
    df=np.array(df)
    return df[:,:-1],df[:,-1]

def R_airfoil():
    # https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise
    df = pd.read_csv(filepath_or_buffer='Datasets/Regression/airfoil.txt', delimiter='\t', header=None)
    df = np.array(df)
    return df[:, :-1], df[:, -1]

def R_housing():
    # https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    df = pd.read_csv(filepath_or_buffer='Datasets/Regression/housing.txt',  header=None, delimiter=r"\s+")
    df = np.array(df)
    return df[:, :-1], df[:, -1]

def R_ccp():
    # https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant
    df = pd.read_csv(filepath_or_buffer='Datasets/Regression/ccp.csv', header=None, delimiter=',')
    df = np.array(df)
    return df[:, :-1], df[:, -1]

def R_superconduct():
    # https://archive.ics.uci.edu/ml/datasets/superconductivty+data
    df = pd.read_csv(filepath_or_buffer='Datasets/Regression/superconduct.csv', header=None, delimiter=',')
    df = np.array(df)
    return df[:, :-1], df[:, -1]

def R_energy():
    # https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
    df = pd.read_csv(filepath_or_buffer='Datasets/Regression/energydata.csv', delimiter=',')
    df.drop('date', axis=1, inplace=True)
    df.drop('rv1', axis=1, inplace=True)
    df.drop('rv2', axis=1, inplace=True)
    print(df.columns)
    df=np.array(df)
    return df[:, 1:], df[:, 0]

