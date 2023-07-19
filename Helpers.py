import os
import numpy as np
import pandas as pd
from joblib import dump
from os import listdir
from os.path import isdir, join
from joblib import dump, load


def save_meta_models(models, dataset, model_name, feature_map, directory):
    file_name= dataset +'_'+ model_name + '_' + feature_map +'.joblib'
    for i, m in enumerate(models['Model']):
        dump(m, directory+'/'+ str(models['Fold_Num'][i])+ '_'+ file_name)

# generate a directory name to write performance metrics and read_me text
def generate_directory_name():
    onlyfiles = [f for f in listdir('.') if isdir(join('.', f))]
    onlyfiles = sorted([int(f.split('Results')[-1].split('_')[-1]) for f in onlyfiles if len(f.split('_')) ==3])

    if len(onlyfiles) > 0:
        return "Numerical_Results_"+str(int(onlyfiles[-1] + 1))
    return "Numerical_Results_"+str(int(0))


# writes the performance metrics obtained in each fold
def save_performance_metrics(dataset, model, feature_map, performance_metrics,directory):
    performance_metrics["Dataset"] = [dataset for i in range(len(performance_metrics['Train_Time']))]
    performance_metrics["Model"] = [model for i in range(len(performance_metrics['Train_Time']))]
    performance_metrics["Feature_Map"] = [feature_map for i in range(len(performance_metrics['Train_Time']))]
    df = pd.DataFrame.from_dict(performance_metrics)
    df.to_csv(path_or_buf= directory + "/Results.csv", mode='a', header=not os.path.exists(directory + "/Results.csv"),
              index=False)

# this function writes the parameters which are used in the experiment
def save_read_me_text(read_me,directory):
    with open(directory+'\ReadMe.txt', 'w') as f:
        for line in read_me:
            f.write(line)
            f.write('\n')

            
def print_array(arr):
    """
    prints a 2-D numpy array in a nicer format
    """
    for a in arr:
        for elem in a:
            print("{}".format(elem).rjust(3), end="")
        print(end="\n")

# formatting peformance scores to latex table format
def format_to_latex_table(performace_scores):
    formatted_results=[[key for key in performace_scores.keys()]]
    print(performace_scores)
    for i in range(len(performace_scores["Dataset"])):
        formatted_results.append([performace_scores[key][i] for key in performace_scores.keys()] )
    formatted_results = np.array(formatted_results)
    averages=["\\textbf{Average}"]
    medians=["\\textbf{Median}"]
    for j in range(1,len(formatted_results[0])):
        averages.append(np.round(np.mean(np.array(formatted_results[1:,j], dtype=float)),4))
        medians.append(np.round(np.median(np.array(formatted_results[1:,j], dtype=float)),4))
    averages=np.array(averages,dtype=str)
    medians = np.array(medians, dtype=str)
    formatted_results= np.insert(formatted_results, range(1,len(formatted_results[0])), "\t\t\t&",axis=1)
    formatted_results= np.hstack((formatted_results,np.array(["\t\t\t\t\\\\" for i in range(len(formatted_results))]).reshape((len(formatted_results),1)) ))
    print_array(formatted_results)
    print("\\midrule")
    print("&".join(averages)+"\t\t\t\t\\\\")
    print("\\midrule")
    print("&".join(medians)+"\t\t\t\t\\\\")

# calculate averages of   given performance scores for a metric obtained with different models, feature maps and datasets
def calc_averages(results, metric, datasets,models, feature_maps):
    for m in models:
        average_performance_metrics = {"Dataset": datasets}
        average_performance_metrics[m]=[]
        for fm in  feature_maps:
            if fm != "Linear":
                average_performance_metrics[fm]=[]
            for dt in datasets:
                filtered=results[metric][(results['Dataset'] == dt)  & (results["Feature_Map"] == fm) & (results["Model"] == m)]
                if fm == "Linear":
                    if  metric == 'Train_Time' or metric=='N_Proto':
                        average_performance_metrics[m].append(np.round(np.mean(filtered), 2))
                    elif  metric=='Test_MSE':

                        average_performance_metrics[m].append(np.round(np.mean(filtered), 4))
                    else:
                        average_performance_metrics[m].append(np.round(100*np.mean(filtered), 2))
                else:
                    if  metric == 'Train_Time' or metric=='N_Proto':
                        average_performance_metrics[fm].append(np.round(np.mean(filtered), 2))
                    elif metric == "Test_MSE":

                        average_performance_metrics[fm].append(np.round(np.mean(filtered), 4))
                    else:
                        average_performance_metrics[fm].append(np.round(100*np.mean(filtered), 2))
        format_to_latex_table(average_performance_metrics)

# reads obtained results from given path
def result_summarizer(path=None):
    results =pd.read_csv(filepath_or_buffer=path, delimiter=',', index_col=None)
    # get unique datasets
    unique_datasets = np.unique(results['Dataset'])
    # get unique models
    unique_models= np.unique(results['Model'])
    # get unique mapping functions
    u, ind = np.unique(results['Feature_Map'], return_index=True)
    unique_feature_maps=u[np.argsort(ind)]
    # get unique metrics
    unique_metrics = np.delete(results.columns.values.flatten(), [0,-1,-2, -3])
    # for each metric calculate average scores
    for metric in unique_metrics:
        print("********************************"+ metric)
        calc_averages(results=results,metric=metric,datasets=unique_datasets,models=unique_models,feature_maps=unique_feature_maps)



#print(result_summarizer(path="FinalResults/California_Housing/Numerical_Results_1/Results.csv"))
print(result_summarizer(path="Numerical_Results_42/Results.csv"))