import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
# Framework:
from preprocessing.preprocessing import convert_categorical_variables, standardize, preprocess_credit_card_data, \
    preprocess_kdd98, preprocess_give_me_some_credit, preprocess_telco_customer_churn, preprocess_default_credit_card, \
    preprocess_bank_marketing
from experiments.experimental_design import experimental_design
from performance_metrics.performance_metrics import get_performance_metrics
from performance_metrics.performance_metrics import evaluate_experiments
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

        if (self.l1 and self.l2):
            raise ValueError('Only l1 or l2 regularization allowed, not both!')

        self.datasets = datasets
        self.methodologies = methodologies
        self.evaluators = evaluators

        self.results_tr_instance = {}
        self.results_tr_class = {}
        self.results_tr_insensitive = {}

    def run(self):
        """
        LOAD AND PREPROCESS DATA
        """
        print('\n\n************** LOADING DATA **************\n')

        # Verify that only one dataset is selected
        if sum(self.datasets.values()) != 1:
            raise ValueError('Select only one dataset!')

        # Kaggle - Credit card data
        if self.datasets['kaggle credit card fraud']:
            print('Kaggle Credit Card Fraud')
            # TODO:
            #  Fixed cost sensitivity analysis?
            fixed_cost = 10
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_credit_card_data(fixed_cost=fixed_cost)
        # KDD98 - Direct Mailing Donations
        elif self.datasets['kdd98']:
            print('KDD98 Direct Mailing Donations')
            covariates, labels, amounts, cost_matrix, categorical_variables = preprocess_kdd98()
        # Kaggle - Give Me Some Credit
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
        else:
            raise Exception('No dataset specified')

        """
        RUN EXPERIMENTS
        """
        print('\n\n***** BUILDING CLASSIFICATION MODELS *****')

        # Prepare the cross-validation procedure
        folds = self.settings['folds']
        repeats = self.settings['repeats']
        rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=2290)
        prepr = experimental_design(labels, amounts)

        # Prepare the evaluation matrices
        n_methodologies = sum(self.methodologies.values())
        for key in self.evaluators.keys():
            if self.evaluators[key]:
                self.results_tr_instance[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_class[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')
                self.results_tr_insensitive[key] = np.empty(shape=(n_methodologies, folds * repeats), dtype='object')

        # Todo: should groups = not be added to rskf.split??? Now y?
        for i, (train_val_index, test_index) in enumerate(rskf.split(covariates, prepr)):
            print('\nCross validation: ' + str(i))

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
                # amounts_train, amounts_val = amounts_train_val[train_index], amounts_train_val[val_index]
                cost_matrix_train, cost_matrix_val = cost_matrix_train_val[train_index, :], cost_matrix_train_val[
                                                                                            val_index, :]

                # Setting: instance or class-dependent costs?
                if self.settings['class_costs']:
                    cost_matrix_train = np.tile(cost_matrix_train.mean(axis=0)[None, :], (len(y_train), 1, 1))
                    cost_matrix_val = np.tile(cost_matrix_val.mean(axis=0)[None, :], (len(y_val), 1, 1))

            # Convert categorical variables + standardization + convert to numpy
            x_train = convert_categorical_variables(x_train, y_train, categorical_variables)
            x_val = convert_categorical_variables(x_val, y_val, categorical_variables)
            x_test = convert_categorical_variables(x_test, y_test, categorical_variables)
            x_train, x_val, x_test = standardize(x_train=x_train, x_val=x_val, x_test=x_test)

            # Assign thresholds:
            #   Instance-dependent cost-sensitive threshold
            threshold_instance = (cost_matrix_test[:, 1, 0] - cost_matrix_test[:, 0, 0]) / (
                                  cost_matrix_test[:, 1, 0] - cost_matrix_test[:, 0, 0]
                                  + cost_matrix_test[:, 0, 1] - cost_matrix_test[:, 1, 1])

            # Todo: can be -np.inf -> What to do?
            #     Use np.divide? What value to set?
            #     threshold_instance = np.maximum(threshold_instance, -100)

            #   Class-dependent cost-sensitive
            #   Todo: can be negative depending on cost matrix -> 0?
            threshold_class = (cost_matrix_test[:, 1, 0].mean() - cost_matrix_test[:, 0, 0].mean()) / (
                               cost_matrix_test[:, 1, 0].mean() - cost_matrix_test[:, 0, 0].mean()
                               + cost_matrix_test[:, 0, 1].mean() - cost_matrix_test[:, 1, 1].mean())
            threshold_class = np.repeat(threshold_class, len(y_test))
            #   Cost-insensitive threshold
            threshold_cost_ins = np.repeat(0.5, len(y_test))

            # Logistic regression (no regularization)
            if self.methodologies['logistic regression']:
                print('\tlogistic regression:')
                # Todo: no warm start? It is a bit weird...
                # Get initial estimate for theta and create model
                init_logit = LogisticRegression(penalty='none', max_iter=2000, verbose=False)
                init_logit.fit(x_train, y_train)
                init_theta = np.insert(init_logit.coef_, 0, values=init_logit.intercept_)

                logit = CSLogit(init_theta, obj='ce')

                # Tune regularization parameters, if necessary
                logit.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train,
                                    cost_matrix_train, x_val, y_val, cost_matrix_val)

                logit.fitting(x_train, y_train, cost_matrix_train)

                logit_proba = logit.predict(x_test)

                logit_pred = (logit_proba > threshold_instance).astype(int)
                self.results_tr_instance = get_performance_metrics(self.evaluators, self.results_tr_instance, i, index,
                                                                   cost_matrix_test, y_test, logit_proba, logit_pred)

                logit_pred = (logit_proba > threshold_class).astype(int)
                self.results_tr_class = get_performance_metrics(self.evaluators, self.results_tr_class, i, index,
                                                                cost_matrix_test, y_test, logit_proba, logit_pred)

                logit_pred = (logit_proba > threshold_cost_ins).astype(int)
                self.results_tr_insensitive = get_performance_metrics(self.evaluators, self.results_tr_insensitive, i,
                                                                      index, cost_matrix_test, y_test, logit_proba,
                                                                      logit_pred)

                index += 1

            if self.methodologies['wlogit']:
                print('\twlogit:')
                try:
                    init_logit
                except NameError:
                    # Todo: make sure the same settings are used!
                    init_logit = LogisticRegression(penalty='none', max_iter=2000, verbose=False)
                    init_logit.fit(x_train, y_train)
                init_theta = np.insert(init_logit.coef_, 0, values=init_logit.intercept_)

                wlogit = CSLogit(init_theta, obj='weightedce')

                # Tune regularization parameters, if necessary
                wlogit.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train,
                                    cost_matrix_train, x_val, y_val, cost_matrix_val)

                wlogit.fitting(x_train, y_train, cost_matrix_train)

                wlogit_proba = wlogit.predict(x_test)

                wlogit_pred = (wlogit_proba > threshold_instance).astype(int)
                self.results_tr_instance = get_performance_metrics(self.evaluators, self.results_tr_instance, i, index,
                                                                   cost_matrix_test, y_test, wlogit_proba, wlogit_pred)

                wlogit_pred = (wlogit_proba > threshold_class).astype(int)
                self.results_tr_class = get_performance_metrics(self.evaluators, self.results_tr_class, i, index,
                                                                cost_matrix_test, y_test, wlogit_proba, wlogit_pred)

                wlogit_pred = (wlogit_proba > threshold_cost_ins).astype(int)
                self.results_tr_insensitive = get_performance_metrics(self.evaluators, self.results_tr_insensitive, i,
                                                                      index, cost_matrix_test, y_test, wlogit_proba,
                                                                      wlogit_pred)

                index += 1

            # Cost-sensitive logistic regression
            if self.methodologies['cslogit']:
                print('\tcslogit:')
                try:
                    init_logit
                except NameError:
                    init_logit = LogisticRegression(penalty='none', max_iter=2000)
                    init_logit.fit(x_train, y_train)
                init_theta = np.insert(init_logit.coef_, 0, values=init_logit.intercept_)

                cslogit = CSLogit(init_theta, obj='aec')

                cslogit.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train,
                                    cost_matrix_train, x_val, y_val, cost_matrix_val)

                cslogit.fitting(x_train, y_train, cost_matrix_train)

                cslogit_proba = cslogit.predict(x_test)

                cslogit_pred = (cslogit_proba > threshold_instance).astype(int)
                self.results_tr_instance = get_performance_metrics(self.evaluators, self.results_tr_instance, i, index,
                                                                   cost_matrix_test, y_test, cslogit_proba,
                                                                   cslogit_pred)

                cslogit_pred = (cslogit_proba > threshold_class).astype(int)
                self.results_tr_class = get_performance_metrics(self.evaluators, self.results_tr_class, i, index,
                                                                cost_matrix_test, y_test, cslogit_proba, cslogit_pred)

                cslogit_pred = (cslogit_proba > threshold_cost_ins).astype(int)
                self.results_tr_insensitive = get_performance_metrics(self.evaluators, self.results_tr_insensitive, i,
                                                                      index, cost_matrix_test, y_test, cslogit_proba,
                                                                      cslogit_pred)

                index += 1

            if self.methodologies['neural network']:
                print('\tneural network:')

                neural_network = CSNeuralNetwork(n_inputs=x_train.shape[1], obj='ce')

                neural_network = neural_network.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list,
                                                     self.neurons_list, x_train, y_train, cost_matrix_train, x_val,
                                                     y_val, cost_matrix_val)

                # Todo: Delete
                print('SUMMARY:')
                print('Number of neurons = {}'.format(neural_network.__getattr__('lin_layer1').out_features))
                print('Lambda 1 = {}'.format(neural_network.lambda1))
                print('Lambda 2 = {}'.format(neural_network.lambda2))

                neural_network = neural_network.model_train(neural_network, x_train, y_train, x_val, y_val,
                                                            cost_matrix_train=cost_matrix_train,
                                                            cost_matrix_val=cost_matrix_val)

                nn_proba = neural_network.model_predict(neural_network, x_test)

                nn_pred = (nn_proba > threshold_instance).astype(int)
                self.results_tr_instance = get_performance_metrics(self.evaluators, self.results_tr_instance, i, index,
                                                                   cost_matrix_test, y_test, nn_proba, nn_pred)

                nn_pred = (nn_proba > threshold_class).astype(int)
                self.results_tr_class = get_performance_metrics(self.evaluators, self.results_tr_class, i, index,
                                                                cost_matrix_test, y_test, nn_proba, nn_pred)

                nn_pred = (nn_proba > threshold_cost_ins).astype(int)
                self.results_tr_insensitive = get_performance_metrics(self.evaluators, self.results_tr_insensitive, i,
                                                                      index, cost_matrix_test, y_test, nn_proba,
                                                                      nn_pred)

                index += 1

            if self.methodologies['wnet']:
                print('\twnet:')

                wnet = CSNeuralNetwork(n_inputs=x_train.shape[1], obj='weightedce')

                wnet = wnet.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, self.neurons_list, x_train,
                                 y_train, cost_matrix_train, x_val, y_val, cost_matrix_val)

                wnet = wnet.model_train(wnet, x_train, y_train, x_val, y_val,
                                        cost_matrix_train=cost_matrix_train, cost_matrix_val=cost_matrix_val)

                wnet_proba = wnet.model_predict(wnet, x_test)

                wnet_pred = (wnet_proba > threshold_instance).astype(int)
                self.results_tr_instance = get_performance_metrics(self.evaluators, self.results_tr_instance, i, index,
                                                                   cost_matrix_test, y_test, wnet_proba, wnet_pred)

                wnet_pred = (wnet_proba > threshold_class).astype(int)
                self.results_tr_class = get_performance_metrics(self.evaluators, self.results_tr_class, i, index,
                                                                cost_matrix_test, y_test, wnet_proba, wnet_pred)

                wnet_pred = (wnet_proba > threshold_cost_ins).astype(int)
                self.results_tr_insensitive = get_performance_metrics(self.evaluators, self.results_tr_insensitive, i,
                                                                      index, cost_matrix_test, y_test, wnet_proba,
                                                                      wnet_pred)

                index += 1

            if self.methodologies['csnet']:
                print('\tcsnet:')

                csnet = CSNeuralNetwork(n_inputs=x_train.shape[1], obj='aec')

                csnet = csnet.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, self.neurons_list, x_train,
                                   y_train, cost_matrix_train, x_val, y_val, cost_matrix_val)

                csnet = csnet.model_train(csnet, x_train, y_train, x_val, y_val,
                                          cost_matrix_train=cost_matrix_train, cost_matrix_val=cost_matrix_val)

                csnet_proba = csnet.model_predict(csnet, x_test)

                csnet_pred = (csnet_proba > threshold_instance).astype(int)
                self.results_tr_instance = get_performance_metrics(self.evaluators, self.results_tr_instance, i, index,
                                                                   cost_matrix_test, y_test, csnet_proba, csnet_pred)

                csnet_pred = (csnet_proba > threshold_class).astype(int)
                self.results_tr_class = get_performance_metrics(self.evaluators, self.results_tr_class, i, index,
                                                                cost_matrix_test, y_test, csnet_proba, csnet_pred)

                csnet_pred = (csnet_proba > threshold_cost_ins).astype(int)
                self.results_tr_insensitive = get_performance_metrics(self.evaluators, self.results_tr_insensitive, i,
                                                                      index, cost_matrix_test, y_test, csnet_proba,
                                                                      csnet_pred)

                index += 1

            if self.methodologies['xgboost']:
                print('\txgboost:')

                xgboost = CSBoost(obj='ce')

                xgboost.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train,
                                    cost_matrix_train, x_val, y_val, cost_matrix_val)

                xgboost = xgboost.fit(x_train, y_train, x_val, y_val)

                xgboost_proba = xgboost.inplace_predict(x_test)

                xgboost_pred = (xgboost_proba > threshold_instance).astype(int)
                self.results_tr_instance = get_performance_metrics(self.evaluators, self.results_tr_instance, i, index,
                                                                   cost_matrix_test, y_test, xgboost_proba,
                                                                   xgboost_pred)

                xgboost_pred = (xgboost_proba > threshold_class).astype(int)
                self.results_tr_class = get_performance_metrics(self.evaluators, self.results_tr_class, i, index,
                                                                cost_matrix_test, y_test, xgboost_proba,
                                                                xgboost_pred)

                xgboost_pred = (xgboost_proba > threshold_cost_ins).astype(int)
                self.results_tr_insensitive = get_performance_metrics(self.evaluators, self.results_tr_insensitive, i,
                                                                      index, cost_matrix_test, y_test, xgboost_proba,
                                                                      xgboost_pred)

                index += 1

            if self.methodologies['wboost']:
                print('\twboost:')

                wboost = CSBoost(obj='weightedce')
                wboost.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train,
                                    cost_matrix_train, x_val, y_val, cost_matrix_val)

                wboost = wboost.fit(x_train, y_train, x_val, y_val, cost_matrix_train, cost_matrix_val)

                wboost_proba = wboost.inplace_predict(x_test)

                wboost_pred = (wboost_proba > threshold_instance).astype(int)
                self.results_tr_instance = get_performance_metrics(self.evaluators, self.results_tr_instance, i, index,
                                                                   cost_matrix_test, y_test, wboost_proba,
                                                                   wboost_pred)

                wboost_pred = (wboost_proba > threshold_class).astype(int)
                self.results_tr_class = get_performance_metrics(self.evaluators, self.results_tr_class, i, index,
                                                                cost_matrix_test, y_test, wboost_proba,
                                                                wboost_pred)

                wboost_pred = (wboost_proba > threshold_cost_ins).astype(int)
                self.results_tr_insensitive = get_performance_metrics(self.evaluators, self.results_tr_insensitive, i,
                                                                      index, cost_matrix_test, y_test, wboost_proba,
                                                                      wboost_pred)

                index += 1

            if self.methodologies['csboost']:
                print('\tcsboost:')

                csboost = CSBoost(obj='aec')

                csboost.tune(self.l1, self.lambda1_list, self.l2, self.lambda2_list, x_train, y_train,
                                    cost_matrix_train, x_val, y_val, cost_matrix_val)

                csboost = csboost.fit(x_train, y_train, x_val, y_val, cost_matrix_train, cost_matrix_val)

                csboost_proba = expit(csboost.inplace_predict(x_test))

                csboost_pred = (csboost_proba > threshold_instance).astype(int)
                self.results_tr_instance = get_performance_metrics(self.evaluators, self.results_tr_instance, i, index,
                                                                   cost_matrix_test, y_test, csboost_proba,
                                                                   csboost_pred)

                csboost_pred = (csboost_proba > threshold_class).astype(int)
                self.results_tr_class = get_performance_metrics(self.evaluators, self.results_tr_class, i, index,
                                                                cost_matrix_test, y_test, csboost_proba, csboost_pred)

                csboost_pred = (csboost_proba > threshold_cost_ins).astype(int)
                self.results_tr_insensitive = get_performance_metrics(self.evaluators, self.results_tr_insensitive, i,
                                                                      index, cost_matrix_test, y_test, csboost_proba,
                                                                      csboost_pred)

                index += 1

            print('\n----------------------------------------------------------------')

    def evaluate(self, directory):
        """
        EVALUATION
        """
        print('\n\n********* EVALUATING CLASSIFIERS *********')

        # Todo: write names of thresholds to files!

        print('\n*** Instance-dependent thresholds ***\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_instance,
                             directory=directory, name='id')

        # Only traditional metrics and savings are influenced by the threshold
        # Omit everything but these for subsequent evaluations
        for i in self.evaluators.keys():
            if (i != 'traditional') and (i != 'savings') and (i != 'LaTeX'):
                self.evaluators[i] = False

        print('\n*** Class-dependent thresholds ***\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_class,
                             directory=directory, name='cd')

        print('\n*** Cost-insensitive thresholds ***\n')
        evaluate_experiments(evaluators=self.evaluators,
                             methodologies=self.methodologies,
                             evaluation_matrices=self.results_tr_insensitive,
                             directory=directory, name='ins')
