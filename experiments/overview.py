# Global overview for all experiments
import json
import datetime
from experiments import experiment

# Set project directory:
DIR = 'C:/Users/u0140754/Google Drive/PhD/Projecten/Cost-sensitive learning/Experiments/'

# TODO:
#  Save figures with tikzplotlib (?) or to pdf
#  Save models?

# Specify experimental configuration
#   l1 and l2 not supported simultaneously!
settings = {'class_costs': False,
            'folds': 5,
            'repeats': 2,
            'val_ratio': 0.25,  # Relative to training set only (excluding test set)
            'l1_regularization': False,
            'lambda1_options': [0, 1e-4, 1e-3, 1e-2, 1e-1],
            'l2_regularization': False,
            'lambda2_options': [0, 1e-4, 1e-3, 1e-2, 1e-1],
            'neurons_options': [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10]  # Add more for final
            }

datasets = {'kaggle credit card fraud': False,
            'kdd98': False,
            'kaggle give me some credit': False,
            'kaggle telco customer churn': True,
            'uci default of credit card clients': False,
            'uci bank marketing': False
            }

methodologies = {'logistic regression': True,  # Todo: rename!
                 'wlogit': True,
                 'cslogit': True,

                 'neural network': True,  # Todo: rename!
                 'wnet': True,
                 'csnet': True,

                 'xgboost': True,
                 'wboost': True,
                 'csboost': True
                 }

evaluators = {'LaTeX': False,

              # Cost-insensitive
              'traditional': True,
              'ROC': True,
              'AUC': True,
              'PR': True,
              'H_measure': True,   # Todo: make faster!
              'brier': True,

              # Cost-sensitive
              'savings': True,
              'ROCIV': True,
              'PRIV': True,
              'rankings': True,

              # Todo: Summarize regularization parameters
              'lambda l1': False,
              'lambda l2': False
              }

# TODO:
#     Total cost / total cost matrix
#     Number of parameters!
#         Effective number of parameters? -> Regularization?
#     Model similarity - Look at predictions? (IoU or just % overlap?)
#     Store all probabilities (and labels)?
#     Add final values of lambda1/lambda2 if relevant!
#     Add final values of number of neurons
#     Logistic regression coefficients
#     Gradient boosting feature importance


if __name__ == '__main__':
    # TODO:
    #   Loop over all datasets

    print('\n' + datetime.datetime.now().strftime('Experiment started at:  %d-%m-%y  |  %H:%M'))

    experiment = experiment.Experiment(settings, datasets, methodologies, evaluators)
    experiment.run()

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

    # TODO:
    #   Same 'random state' parameter everywhere
