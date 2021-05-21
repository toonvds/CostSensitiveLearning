# Global overview for all experiments
import os
import json
import datetime
from experiments import experiment

# Set project directory:
DIR = 'C:/Users/u0140754/Google Drive/PhD/Projecten/Cost-sensitive learning/Experiments/'
if not os.path.isdir(DIR):  # Switch to HPC directory
    DIR = '/data/leuven/341/vsc34195/Experiments/Tests'  # TODO: change per experiment!!
if not os.path.isdir(DIR):  # Switch to Google Colab
    DIR = '/content/drive/My Drive/PhD/Projecten/Cost-sensitive learning/Experiments/Colab/'
assert os.path.isdir(DIR), "DIR does not exist!"

import torch
print('CUDA available?')
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# Specify experimental configuration
#   l1 and l2 not supported simultaneously!
settings = {'class_costs': False,
            'folds': 5,
            'repeats': 2,
            'val_ratio': 0.25,  # Relative to training set only (excluding test set)
            'l1_regularization': False,
            'lambda1_options': [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'l2_regularization': False,
            'lambda2_options': [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'neurons_options': [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10]  # Add more for final?
            }

datasets = {'kaggle credit card fraud': False,
            'kdd98 direct mailing': False,
            'kaggle give me some credit': False,
            'kaggle telco customer churn': False,
            'uci default of credit card clients': False,
            'uci bank marketing': False,
            'vub credit scoring': False,
            'tv subscription churn': False,
            'kaggle ieee fraud': True
            }

methodologies = {'logit': True,
                 'wlogit': True,
                 'cslogit': True,

                 'net': True,
                 'wnet': True,
                 'csnet': True,

                 'boost': True,
                 'wboost': True,
                 'csboost': True
                 }

evaluators = {'LaTeX': False,

              # Cost-insensitive
              'traditional': True,
              'ROC': True,
              'AUC': True,
              'PR': True,
              'H_measure': True,
              'brier': True,
              'recall_overlap': True,
              'recall_correlation': True,

              # Cost-sensitive
              'savings': True,
              'AEC': True,
              'ROCIV': True,
              'PRIV': True,
              'rankings': True,

              # Other
              'time': True,
              'lambda1': True,
              'lambda2': True,
              'n_neurons': True
              }


if __name__ == '__main__':
    print('\n' + datetime.datetime.now().strftime('Experiment started at:  %d-%m-%y  |  %H:%M'))

    experiment = experiment.Experiment(settings, datasets, methodologies, evaluators)
    experiment.run(directory=DIR)

    # Create txt file for summary of results
    with open(str(DIR + 'summary.txt'), 'w') as file:
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

    experiment.evaluate(directory=DIR)
