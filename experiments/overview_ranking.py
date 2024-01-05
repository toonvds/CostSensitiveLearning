# Global overview for all experiments
import os
import json
import datetime
import sys
from experiments import experiment


# Possible datasets:
names = {'kaggle credit card fraud': 'KCCF',
         'kdd98 direct mailing': 'KDD',
         'kaggle give me some credit': 'GMSC',
         'kaggle telco customer churn': 'KTCC',
         'uci default of credit card clients': 'DCCC',
         'uci bank marketing': 'UBM',
         'vub credit scoring': 'VCS',
         'tv subscription churn': 'TSC',
         'kaggle ieee fraud': 'KIFD',
         'apate credit card fraud': 'ACCF',
         'home equity': 'HMEQ',
         'uk credit scoring': 'UK',
         'bene1 credit scoring': 'BN1',
         'bene2 credit scoring': 'BN2'
         }


# Set project directory:
DIR = 'C:/Users/u0140754/Google Drive/PhD/Projecten/CS Learning to rank/Experiments/'
if not os.path.isdir(DIR):
    # Switch to HPC directory
    # Select dataset from command line if specified:
    print(sys.argv, len(sys.argv))
    if len(sys.argv) > 1:
        inv_names = {v: k for k, v in names.items()}
        dataset = inv_names[str(sys.argv[1])]
    DIR = '/data/leuven/341/vsc34195/ExperimentsRanking/' + str(sys.argv[1]) + '/'

    print(DIR)
# if not os.path.isdir(DIR):  # Switch to Google Colab
#     DIR = '/content/drive/My Drive/PhD/Projecten/Cost-sensitive learning/Experiments/Colab/'
assert os.path.isdir(DIR), "DIR does not exist!"


# Specify experimental configuration
settings = {'folds': 3,
            'repeats': 1,
            'val_ratio': 0.5   # 0.25  # Relative to training set only (excluding test set)
            }

datasets = {'kaggle credit card fraud': False,
            'kdd98 direct mailing': False,
            'kaggle give me some credit': True,
            'kaggle telco customer churn': False,
            'uci default of credit card clients': False,
            'uci bank marketing': False,
            'vub credit scoring': False,
            'tv subscription churn': False,
            'kaggle ieee fraud': False,
            'apate credit card fraud': False,
            'home equity': False,
            'uk credit scoring': False,
            'bene1 credit scoring': False,
            'bene2 credit scoring': False,
            }

# If command line argument is given, select the right dataset
try:
    dataset
    for key in datasets.keys():
        if key == dataset:
            datasets[key] = True
        else:
            datasets[key] = False
except NameError:
    print('Command line argument not provided. Continuing with specified dataset.')


methodologies = {'xgboost': True,
                 'xgboost_distance': False,
                 'wboost': False,
                 'wboost_distance': False,
                 'csboost': True,
                 'csboost_distance': False,
                 'lambdaMART': False,
                 'lambdaMART_map': False,
                 'lambdaMART_distance': False,
                 'lambdaMART_custom': True,
                 'cslambdaMART': False,
                 'cslambdaMART_cost_matrix': False,
                 'cslambdaMART_distance': False,
                 'cslambdaMART_custom': True,
                 'pyltr': False,
                 'xgboost_top10': False,
                 'xgboost_top100': False,
                 'xgboost_top500': False,
                 'xgboost_top1000': False,
                 'xgboost_top2000': False,
                 'xgboost_top_CI': False
                 }

evaluators = {
    # 'LaTeX': False,         # Not yet implemented

    'spearman_rho': True,
    'AP': True,
    # 'CE': False, # Not yet implemented - very similar to AP

    'top10': True,
    'top100': True,
    'top500': True,
    'top1000': True,
    'top100costs': False,
    'top500costs': False,
    'top1000costs': False,
    'top10profit': True,
    'top100profit': True,
    'top500profit': True,
    'top1000profit': True,
    'cumulative_recovered': False,
    'cumulative_profit': True,
    'expected_precision': True,
    'expected_profit': True
}

if __name__ == '__main__':
    print('\n' + datetime.datetime.now().strftime('Experiment started at:  %d-%m-%y  |  %H:%M'))

    experiment = experiment.RankingExperiment(settings, datasets, methodologies, evaluators)
    start = datetime.datetime.now()
    experiment.run(directory=DIR)
    end = datetime.datetime.now()
    print(end - start)

    dataset = [selected for selected in datasets.keys() if datasets[selected]]

    name = names[dataset[0]]

    # Create txt file for summary of results
    with open(str(DIR + 'summary_' + name + '.txt'), 'w') as file:
        file.write(str(datetime.datetime.now().strftime('Experiment done at:  %d-%m-%y  |  %H:%M') + '\n'))
        file.write('\nSettings: ')
        file.write(json.dumps(settings, indent=3))
        file.write('\nDatasets: ')
        file.write(json.dumps(datasets, indent=3))
        file.write('\nMethodologies: ')
        file.write(json.dumps(methodologies, indent=3))
        file.write('\nEvaluators: ')
        file.write(json.dumps(evaluators, indent=3))
        file.write('\n\n_____________________________________________________________________\n\n')

    DIR = str(DIR + 'summary_' + name + '.txt')

    experiment.evaluate(directory=DIR)
