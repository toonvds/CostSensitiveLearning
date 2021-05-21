import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import time
# Framework:
from preprocessing.preprocessing import convert_categorical_variables, standardize, handle_missing_data, \
    preprocess_credit_card_data, preprocess_kdd98, preprocess_give_me_some_credit, preprocess_telco_customer_churn, \
    preprocess_default_credit_card, preprocess_bank_marketing, preprocess_vub_credit_scoring, \
    preprocess_tv_subscription_churn, preprocess_kaggle_ieee_fraud
from experiments.experimental_design import experimental_design
from performance_metrics.performance_metrics import get_performance_metrics, evaluate_experiments, cost_with_algorithm
# Models:
from methodologies.cs_logit.cs_logit import CSLogit
from methodologies.cs_net import CSNeuralNetwork
from methodologies.cs_boost import CSBoost


class Experiment:
    def __init__(self, settings, datasets, methodologies, evaluators):
        self.settings = settings

        self.l1 = self.settings['l1_regularization']
        self.lambda1_list = self.settings['lambda1_options']
        self.l2 = self.settings['l2_regularization']
        self.lambda2_list = self.settings['lambda2_options']
        self.neurons_list = self.settings['neurons_options']

        if self.l1 and self.l2:
            raise ValueError('Only l1 or l2 regularization allowed, not both!')

        self.datasets = datasets
        self.methodologies = methodologies
        self.evaluators = evaluators

        self.results_tr_instance = {}
        self.results_tr_instance_calibrated = {}
        self.results_tr_class = {}
        self.results_tr_class_calibrated = {}
        self.results_tr_class_imb = {}
        self.results_tr_empirical_id = {}
        self.results_tr_empirical_cd = {}
        self.results_tr_empirical_f1 = {}
        self.results_tr_insensitive = {}

    def run(self, directory):
        """
        LOAD AND PREPROCESS DATA
        """
        print('\n\n************** LOADING DATA **************\n')

        # Verify that only one dataset is selected
        if sum(self.datasets.values()) != 1:
            raise ValueError('Select only one dataset!')

        if self.datasets['kaggle credit card fraud']:
            print('Kaggle Credit Card Fraud')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_credit_card_data(fixed_cost=10)
        elif self.datasets['kdd98 direct mailing']:
            print('KDD98 Direct Mailing Donations')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_kdd98()
        elif self.datasets['kaggle give me some credit']:
            print('Kaggle Give Me Some Credit')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_give_me_some_credit()
        elif self.datasets['kaggle telco customer churn']:
            print('Kaggle Telco Customer Churn')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_telco_customer_churn()
        elif self.datasets['uci default of credit card clients']:
            print('UCI Default of Credit Card Clients')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_default_credit_card()
        elif self.datasets['uci bank marketing']:
            print('UCI Bank Marketing')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_bank_marketing()
        elif self.datasets['vub credit scoring']:
            print('VUB Credit Scoring')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_vub_credit_scoring()
        elif self.datasets['tv subscription churn']:
            print('TV Subscription Churn')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_tv_subscription_churn()
        elif self.datasets['kaggle ieee fraud']:
            print('Kaggle IEEE Fraud Detection')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_kaggle_ieee_fraud(subsample=1)
        else:
            raise Exception('No dataset specified')

        """
        RUN EXPERIMENTS
        """
        print('\n\n***** BUILDING CLASSIFICATION MODELS *****')

        # Prepare the cross-validation procedure
        folds = self.settings['folds']
        repeats = self.settings['repeats']
        rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
        prepr = experimental_design(labels, amounts)

        # Prepare the evaluation matrices
        n_methodologies = sum(self.methodologies.values())
        for key in self.evaluators.keys():
            if self.evaluators[key]:
                self.results_tr_instance[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_instance_calibrated[key] = np.empty(shape=(n_methodologies, folds * repeats),
                                                                    dtype='object')
                self.results_tr_class[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_class_calibrated[key] = np.empty(shape=(n_methodologies, folds * repeats),
                                                                 dtype='object')
                self.results_tr_class_imb[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_empirical_id[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_empirical_cd[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_empirical_f1[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_insensitive[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')

        for i, (train_val_index, test_index) in enumerate(rskf.split(covariates, prepr)):
            print('\nCross validation: ' + str(i + 1))

            index = 0

            x_train_val, x_test = covariates.iloc[train_val_index], covariates.iloc[test_index]
            y_train_val, y_test = labels[train_val_index], labels[test_index]
            amounts_train_val, amounts_test = amounts[train_val_index], amounts[test_index]
            cost_matrix_train_val, cost_matrix_test = cost_matrix[train_val_index, :], cost_matrix[test_index, :]

            # Split training and validation set (based on instance-dependent costs)
            train_ratio = 1 - self.settings['val_ratio']
            skf = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=42)
            prepr_val = experimental_design(y_train_val, amounts_train_val)

            for train_index, val_index in skf.split(x_train_val, prepr_val):
                x_train, x_val = x_train_val.iloc[train_index], x_train_val.iloc[val_index]
                y_train, y_val = y_train_val[train_index], y_train_val[val_index]
                cost_matrix_train, cost_matrix_val = cost_matrix_train_val[train_index, :], cost_matrix_train_val[
                                                                                            val_index, :]

                # Setting: instance or class-dependent costs?
                if self.settings['class_costs']:
                    cost_matrix_train = np.tile(cost_matrix_train.mean(axis=0)[None, :], (len(y_train), 1, 1))
                    cost_matrix_val = np.tile(cost_matrix_val.mean(axis=0)[None, :], (len(y_val), 1, 1))

            # Preprocessing: Handle missing data, convert categorical variables, standardize, convert to numpy
            x_train, x_val, x_test, categorical_variables = handle_missing_data(x_train, x_val, x_test,
                                                                                categorical_variables)
            x_train, x_val, x_test = convert_categorical_variables(x_train, y_train, x_val, x_test,
                                                                   categorical_variables)
            x_train, x_val, x_test = standardize(x_train=x_train, x_val=x_val, x_test=x_test)

            # Assign thresholds for the different strategies:
            #   Instance-dependent cost-sensitive threshold
            threshold_instance = (cost_matrix_test[:, 1, 0] - cost_matrix_test[:, 0, 0]) / (
                                  cost_matrix_test[:, 1, 0] - cost_matrix_test[:, 0, 0]
                                  + cost_matrix_test[:, 0, 1] - cost_matrix_test[:, 1, 1])

            #   Class-dependent cost-sensitive threshold
            threshold_class = (cost_matrix_test[:, 1, 0].mean() - cost_matrix_test[:, 0, 0].mean()) / (
                               cost_matrix_test[:, 1, 0].mean() - cost_matrix_test[:, 0, 0].mean()
                               + cost_matrix_test[:, 0, 1].mean() - cost_matrix_test[:, 1, 1].mean())
            threshold_class = np.repeat(threshold_class, len(y_test))
            #   Class imbalance threshold
            threshold_class_imbalance = y_train.mean()
            #   Cost-insensitive threshold
            threshold_cost_ins = np.repeat(0.5, len(y_test))

            # Define evaluation procedure for different thresholding strategies
            def evaluate_model(proba_val, proba, j, index, info):
                # ID CS Threshold:
                pred = (proba > threshold_instance).astype(int)
                self.results_tr_instance = get_performance_metrics(self.evaluators, self.results_tr_instance, j, index,
                                                                   cost_matrix_test, y_test, proba, pred, info)

                # ID CS Threshold with calibrated probabilities (using isotonic regression):
                isotonic = IsotonicRegression(out_of_bounds='clip')
                isotonic.fit(proba_val, y_val)     # Fit on validation set!
                proba_calibrated = isotonic.transform(proba)

                pred = (proba_calibrated > threshold_instance).astype(int)
                self.results_tr_instance_calibrated = get_performance_metrics(self.evaluators,
                                                                              self.results_tr_instance_calibrated, j,
                                                                              index, cost_matrix_test, y_test,
                                                                              proba_calibrated, pred, info)

                # CD CS Threshold:
                pred = (proba > threshold_class).astype(int)
                self.results_tr_class = get_performance_metrics(self.evaluators, self.results_tr_class, j, index,
                                                                cost_matrix_test, y_test, proba, pred, info)

                # CD CS Threshold with calibrated probabilities:
                pred = (proba_calibrated > threshold_class).astype(int)
                self.results_tr_class_calibrated = get_performance_metrics(self.evaluators,
                                                                           self.results_tr_class_calibrated, j, index,
                                                                           cost_matrix_test, y_test, proba, pred, info)

                # Class imbalance Threshold:
                pred = (proba > threshold_class_imbalance).astype(int)
                self.results_tr_class_imb = get_performance_metrics(self.evaluators, self.results_tr_class_imb, j,
                                                                    index, cost_matrix_test, y_test, proba, pred, info)

                # Empirical thresholding: ID costs
                threshold_opt_val = empirical_thresholding(proba_val, y_val, cost_matrix_val, metric='idcosts')
                pred = (proba > threshold_opt_val).astype(int)
                self.results_tr_empirical_id = get_performance_metrics(self.evaluators, self.results_tr_empirical_id, j,
                                                                       index, cost_matrix_test, y_test, proba, pred,
                                                                       info)

                # Empirical thresholding: CD costs
                threshold_opt_val = empirical_thresholding(proba_val, y_val, cost_matrix_val, metric='cdcosts')
                pred = (proba > threshold_opt_val).astype(int)
                self.results_tr_empirical_cd = get_performance_metrics(self.evaluators, self.results_tr_empirical_cd, j,
                                                                       index, cost_matrix_test, y_test, proba, pred,
                                                                       info)

                # Empirical thresholding: F1
                threshold_opt_val = empirical_thresholding(proba_val, y_val, cost_matrix_val, metric='f1')
                pred = (proba > threshold_opt_val).astype(int)
                self.results_tr_empirical_f1 = get_performance_metrics(self.evaluators, self.results_tr_empirical_f1,
                                                                       j, index, cost_matrix_test, y_test, proba, pred,
                                                                       info)

                # Cost-insensitive threshold:
                pred = (proba > threshold_cost_ins).astype(int)
                self.results_tr_insensitive = get_performance_metrics(self.evaluators, self.results_tr_insensitive, j,
                                                                      index, cost_matrix_test, y_test, proba, pred,
                                                                      info)

            # Logistic regression
            if self.methodologies['logit']:
                print('\tlogistic regression:')
                # Get initial estimate for theta and create model
                init_logit = LogisticRegression(penalty='none', max_iter=1, verbose=0, solver='sag', n_jobs=-1)
                init_logit.fit(x_train, y_train)
                init_theta = np.insert(init_logit.coef_, 0, values=init_logit.intercept_)

                logit = CSLogit(init_theta, obj='ce')

                # Tune regularization parameters, if necessary
                logit.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train, cost_matrix_train,
                           x_val, y_val, cost_matrix_val)

                lambda1 = logit.lambda1
                lambda2 = logit.lambda2

                start = time.perf_counter()
                logit.fitting(x_train, y_train, cost_matrix_train)
                end = time.perf_counter()

                logit_proba = logit.predict(x_test)
                logit_proba_val = logit.predict(x_val)

                info = {'time': end - start, 'lambda1': lambda1, 'lambda2': lambda2, 'n_neurons': 0}

                evaluate_model(logit_proba_val, logit_proba, i, index, info)

                index += 1

            # Weighted logistic regression
            if self.methodologies['wlogit']:
                print('\twlogit:')
                try:
                    init_logit
                except NameError:
                    init_logit = LogisticRegression(penalty='none', max_iter=1, verbose=False, solver='sag', n_jobs=-1)
                    init_logit.fit(x_train, y_train)
                    init_theta = np.insert(init_logit.coef_, 0, values=init_logit.intercept_)

                wlogit = CSLogit(init_theta, obj='weightedce')

                # Tune regularization parameters, if necessary
                wlogit.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train, cost_matrix_train,
                            x_val, y_val, cost_matrix_val)

                lambda1 = wlogit.lambda1
                lambda2 = wlogit.lambda2

                start = time.perf_counter()
                wlogit.fitting(x_train, y_train, cost_matrix_train)
                end = time.perf_counter()

                wlogit_proba = wlogit.predict(x_test)
                wlogit_proba_val = wlogit.predict(x_val)

                info = {'time': end - start, 'lambda1': lambda1, 'lambda2': lambda2, 'n_neurons': 0}

                evaluate_model(wlogit_proba_val, wlogit_proba, i, index, info)

                index += 1

            # Cost-sensitive logistic regression
            if self.methodologies['cslogit']:
                print('\tcslogit:')
                try:
                    init_logit
                except NameError:
                    init_logit = LogisticRegression(penalty='none', max_iter=1, verbose=False, solver='sag',
                                                    n_jobs=-1)
                    init_logit.fit(x_train, y_train)
                    init_theta = np.insert(init_logit.coef_, 0, values=init_logit.intercept_)

                cslogit = CSLogit(init_theta, obj='aec')

                cslogit.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train,
                                    cost_matrix_train, x_val, y_val, cost_matrix_val)

                lambda1 = cslogit.lambda1
                lambda2 = cslogit.lambda2

                start = time.perf_counter()
                cslogit.fitting(x_train, y_train, cost_matrix_train)
                end = time.perf_counter()

                cslogit_proba = cslogit.predict(x_test)
                cslogit_proba_val = cslogit.predict(x_val)

                info = {'time': end - start, 'lambda1': lambda1, 'lambda2': lambda2, 'n_neurons': 0}

                evaluate_model(cslogit_proba_val, cslogit_proba, i, index, info)

                index += 1

            if self.methodologies['net']:
                print('\tneural network:')

                neural_network = CSNeuralNetwork(n_inputs=x_train.shape[1], obj='ce', directory=directory)

                neural_network = neural_network.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list,
                                                     self.neurons_list, x_train, y_train, cost_matrix_train, x_val,
                                                     y_val, cost_matrix_val)

                lambda1 = neural_network.lambda1
                lambda2 = neural_network.lambda2

                start = time.perf_counter()
                neural_network = neural_network.model_train(neural_network, x_train, y_train, x_val, y_val,
                                                            cost_matrix_train=cost_matrix_train,
                                                            cost_matrix_val=cost_matrix_val)
                end = time.perf_counter()

                nn_proba = neural_network.model_predict(neural_network, x_test)
                nn_proba_val = neural_network.model_predict(neural_network, x_val)

                info = {'time': end - start, 'lambda1': lambda1, 'lambda2': lambda2,
                        'n_neurons': neural_network.lin_layer1.out_features}

                evaluate_model(nn_proba_val, nn_proba, i, index, info)

                index += 1

            if self.methodologies['wnet']:
                print('\twnet:')

                wnet = CSNeuralNetwork(n_inputs=x_train.shape[1], obj='weightedce', directory=directory)

                wnet = wnet.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, self.neurons_list, x_train,
                                 y_train, cost_matrix_train, x_val, y_val, cost_matrix_val)

                lambda1 = wnet.lambda1
                lambda2 = wnet.lambda2

                start = time.perf_counter()
                wnet = wnet.model_train(wnet, x_train, y_train, x_val, y_val,
                                        cost_matrix_train=cost_matrix_train, cost_matrix_val=cost_matrix_val)
                end = time.perf_counter()

                wnet_proba = wnet.model_predict(wnet, x_test)
                wnet_proba_val = wnet.model_predict(wnet, x_val)

                info = {'time': end - start, 'lambda1': lambda1, 'lambda2': lambda2,
                        'n_neurons': wnet.lin_layer1.out_features}

                evaluate_model(wnet_proba_val, wnet_proba, i, index, info)

                index += 1

            if self.methodologies['csnet']:
                print('\tcsnet:')

                csnet = CSNeuralNetwork(n_inputs=x_train.shape[1], obj='aec', directory=directory)

                csnet = csnet.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, self.neurons_list, x_train,
                                   y_train, cost_matrix_train, x_val, y_val, cost_matrix_val)

                lambda1 = csnet.lambda1
                lambda2 = csnet.lambda2

                start = time.perf_counter()
                csnet = csnet.model_train(csnet, x_train, y_train, x_val, y_val,
                                          cost_matrix_train=cost_matrix_train, cost_matrix_val=cost_matrix_val)
                end = time.perf_counter()

                csnet_proba = csnet.model_predict(csnet, x_test)
                csnet_proba_val = csnet.model_predict(csnet, x_val)

                info = {'time': end - start, 'lambda1': lambda1, 'lambda2': lambda2,
                        'n_neurons': csnet.lin_layer1.out_features}

                evaluate_model(csnet_proba_val, csnet_proba, i, index, info)

                index += 1

            if self.methodologies['boost']:
                print('\txgboost:')

                xgboost = CSBoost(obj='ce')

                xgboost.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train,
                                    cost_matrix_train, x_val, y_val, cost_matrix_val)

                lambda1 = xgboost.lambda1
                lambda2 = xgboost.lambda2

                start = time.perf_counter()
                xgboost = xgboost.fit(x_train, y_train, x_val, y_val)
                end = time.perf_counter()

                xgboost_proba = xgboost.inplace_predict(x_test)
                xgboost_proba_val = xgboost.inplace_predict(x_val)

                info = {'time': end-start, 'lambda1': lambda1, 'lambda2': lambda2, 'n_neurons': 0}

                evaluate_model(xgboost_proba_val, xgboost_proba, i, index, info)

                index += 1

            if self.methodologies['wboost']:
                print('\twboost:')

                wboost = CSBoost(obj='weightedce')
                wboost.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train,
                                    cost_matrix_train, x_val, y_val, cost_matrix_val)

                lambda1 = wboost.lambda1
                lambda2 = wboost.lambda2

                start = time.perf_counter()
                wboost = wboost.fit(x_train, y_train, x_val, y_val, cost_matrix_train, cost_matrix_val)
                end = time.perf_counter()

                wboost_proba = wboost.inplace_predict(x_test)
                wboost_proba_val = wboost.inplace_predict(x_val)

                info = {'time': end - start, 'lambda1': lambda1, 'lambda2': lambda2, 'n_neurons': 0}

                evaluate_model(wboost_proba_val, wboost_proba, i, index, info)

                index += 1

            if self.methodologies['csboost']:
                print('\tcsboost:')

                csboost = CSBoost(obj='aec')

                csboost.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train,
                                    cost_matrix_train, x_val, y_val, cost_matrix_val)

                lambda1 = csboost.lambda1
                lambda2 = csboost.lambda2

                start = time.perf_counter()
                csboost = csboost.fit(x_train, y_train, x_val, y_val, cost_matrix_train, cost_matrix_val)
                end = time.perf_counter()

                csboost_proba = expit(csboost.inplace_predict(x_test))
                csboost_proba_val = expit(csboost.inplace_predict(x_val))

                info = {'time': end - start, 'lambda1': lambda1, 'lambda2': lambda2, 'n_neurons': 0}

                evaluate_model(csboost_proba_val, csboost_proba, i, index, info)

                index += 1

            print('\n----------------------------------------------------------------')

    def evaluate(self, directory):
        """
        EVALUATION
        """
        print('\n\n********* EVALUATING CLASSIFIERS *********')

        with open(str(directory + 'summary.txt'), 'a') as file:
            file.write('\n*** Instance-dependent thresholds ***\n')
        print('\n*** Instance-dependent thresholds ***\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_instance,
                             directory=directory, name='id')

        # Do not recalculate constant metrics: time, lambdas
        for key in ['time', 'lambda1', 'lambda2', 'n_neurons']:
            if self.evaluators[key]:
                self.evaluators[key] = False

        with open(str(directory + 'summary.txt'), 'a') as file:
            file.write('\n*** Instance-dependent thresholds with calibrated probabilities ***\n')
        print('\n*** Instance-dependent thresholds with calibrated probabilities ***\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_instance_calibrated,
                             directory=directory, name='id_cal')

        # Only some metrics are influenced by the threshold
        # Omit everything but these for subsequent evaluations
        for i in self.evaluators.keys():
            if (i != 'traditional') and (i != 'savings') and (i != 'recall_overlap') and (i != 'LaTeX'):
                self.evaluators[i] = False

        with open(str(directory + 'summary.txt'), 'a') as file:
            file.write('\n*** Class-dependent thresholds ***\n')
        print('\n*** Class-dependent thresholds ***\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_class,
                             directory=directory, name='cd')

        with open(str(directory + 'summary.txt'), 'a') as file:
            file.write('\n*** Class-dependent thresholds with calibrated probabilities ***\n')
        print('\n*** Class-dependent thresholds with calibrated probabilities ***\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_class_calibrated,
                             directory=directory, name='cd_cal')

        with open(str(directory + 'summary.txt'), 'a') as file:
            file.write('\n*** Empirical thresholding ID ***\n')
        print('\n*** Empirical thresholding ID ***\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_empirical_id,
                             directory=directory, name='emp_id')

        with open(str(directory + 'summary.txt'), 'a') as file:
            file.write('\n*** Empirical thresholding CD ***\n')
        print('\n*** Empirical thresholding CD ***\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_empirical_cd,
                             directory=directory, name='emp_cd')

        with open(str(directory + 'summary.txt'), 'a') as file:
            file.write('\n*** Empirical thresholding F1 ***\n')
        print('\n*** Empirical thresholding F1 ***\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_empirical_f1,
                             directory=directory, name='emp_f1')

        with open(str(directory + 'summary.txt'), 'a') as file:
            file.write('\n*** Class imbalance thresholds ***\n')
        print('\n*** Class imbalance thresholds ***\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_class_imb,
                             directory=directory, name='imb')

        with open(str(directory + 'summary.txt'), 'a') as file:
            file.write('\n*** Cost-insensitive thresholds ***\n')
        print('\n*** Cost-insensitive thresholds ***\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_insensitive,
                             directory=directory, name='ins')


def empirical_thresholding(proba_val, y_val, cost_matrix_val, metric='idcosts'):

    total_costs = []
    thresholds = list(np.unique(proba_val))
    if 0. not in thresholds:
        thresholds = [0.] + thresholds
    if 1. not in thresholds:
        thresholds = thresholds + [1.]

    if metric == 'idcosts':
        for tr in thresholds:
            pred = (proba_val > tr).astype(int)
            total_costs.append(cost_with_algorithm(cost_matrix=cost_matrix_val, labels=y_val, predictions=pred))

    elif metric == 'cdcosts':
        cost_matrix_val = np.tile(cost_matrix_val.mean(axis=0)[None, :], (len(y_val), 1, 1))
        for tr in thresholds:
            pred = (proba_val > tr).astype(int)
            total_costs.append(cost_with_algorithm(cost_matrix=cost_matrix_val, labels=y_val, predictions=pred))

    elif metric == 'f1':
        for tr in thresholds:
            pred = (proba_val > tr).astype(int)
            total_costs.append(f1_score(y_val, pred))

    # Choose lowest threshold for costs:
    if metric == 'idcosts' or metric == 'cdcosts':
        total_costs = np.array(total_costs)
        optimal_index = np.where(total_costs == total_costs.min())[0]
    # Choose highest threshold for f1:
    if metric == 'f1':
        total_costs = np.array(total_costs)
        optimal_index = np.where(total_costs == total_costs.max())[0]

    if len(optimal_index) > 1:
        # If there are multiple minima, choose the one with the widest valley (see Sheng and Ling, 2006, AAAI)
        # In practice, we check the steepness of the valley on both sides (first derivative)
        steepnesses = []
        for ind in optimal_index:
            if not thresholds[ind] == 0:
                steepness_left = (total_costs[ind - 1] - total_costs[ind]) / (thresholds[ind] - thresholds[ind - 1])
            else:
                # Just take the steepness on the right twice
                steepness_left = (total_costs[ind + 1] - total_costs[ind]) / (thresholds[ind + 1] - thresholds[ind])
            if not thresholds[ind] == 1:
                steepness_right = (total_costs[ind + 1] - total_costs[ind]) / (thresholds[ind + 1] - thresholds[ind])
            else:
                # Just take the steepness on the left twice
                steepness_right = (total_costs[ind - 1] - total_costs[ind]) / (thresholds[ind] - thresholds[ind - 1])
            steepness = (steepness_left + steepness_right) / 2
            steepnesses.append(steepness)
        optimal_index = optimal_index[np.argmin(steepness)]
    else:
        optimal_index = optimal_index[0]

    optimal_threshold = thresholds[optimal_index]

    # plt.title(str(metric))
    # plt.plot(thresholds, total_costs)
    # plt.vlines(optimal_threshold, 0, max(total_costs), linestyles='dashed')
    # plt.show()
    #
    # print(str(metric))
    # print(f'Optimal threshold = {optimal_threshold}')

    return optimal_threshold