import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from methodologies.cs_logit import cs_logit
from experiments import experimental_design
from performance_metrics import performance_metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


# repeatedstratifiedKfold generate multiple test
# sets such that all contain the same distribution of classes, or as close as possible.

def cv_plot(covariates, clas, amount, fixed_cost, lambd, cost_matrix, fold, repeats):
    rskf = RepeatedStratifiedKFold(n_splits=fold, n_repeats=repeats, random_state=2290)
    ratio_cost_matrix = np.empty(shape=(3, 2 * repeats), dtype='object')
    prepr = experimental_design.experimental_design(clas, amount)

    i = 0

    for train_index, test_index in rskf.split(covariates, prepr):
        print('Cross Validation number ' + str(i))

        X_train, X_test = covariates[train_index], covariates[test_index]
        y_train, y_test = clas[train_index], clas[test_index]
        amount_train, amount_test = amount[train_index], amount[test_index]
        cost_matrix_sample = cost_matrix.iloc[test_index, :]


        ratio_logist = cv_logit_ratio(fixed_cost, X_train, y_train,
                                      X_test, y_test, amount_test, cost_matrix_sample)

        ratio_cost_matrix[0, i] = 1 - ratio_logist

        ratio_cslogist = cv_cslogit_ratio(lambd, fixed_cost, X_train, y_train, X_test,
                                          y_test, amount_train, amount_test, cost_matrix_sample)

        ratio_cost_matrix[1, i] = 1 - ratio_cslogist

        ratio_cslogist_over = cv_oversample_cslogit_ratio(lambd, fixed_cost, X_train, y_train, X_test,
                                                          y_test, amount_test, cost_matrix_sample)

        ratio_cost_matrix[2, i] = 1 - ratio_cslogist_over

        i += 1

    ratio_cost_matrix = pd.DataFrame({'Logit': ratio_cost_matrix[0, :],
                                      'Cslogit': ratio_cost_matrix[1, :],
                                      'Cslogit_over': ratio_cost_matrix[2, :]})
    print(ratio_cost_matrix)
    print("amount of savings logit (%) " + str(np.mean(ratio_cost_matrix.iloc[:,0])))
    print("amount of savings cs logit (%) " + str(np.mean(ratio_cost_matrix.iloc[:,1])))
    print("amount of savings cs logit oversample (%) " + str(np.mean(ratio_cost_matrix.iloc[:,2])))


    plt.xlabel('Methods')
    plt.ylabel('Savings (%)')
    sns.boxplot(data=ratio_cost_matrix)
    plt.show()


def cv_logit_ratio(fixed_cost, X_train, y_train, X_test, y_test, amount_test, cost_matrix_sample):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    lgtr = LogisticRegression(penalty='none', max_iter=250)
    #lgtr = LogisticRegression(penalty='l2')
    #lgtr = LogisticRegression(penalty='l1', solver='saga', max_iter=250)
    lgtr.fit(X_train, y_train)

    predictions = lgtr.predict_proba(X_test)
    threshold = fixed_cost / amount_test
    predictions = np.where(predictions[:, 1] > threshold, 1, 0)
    cost_with_alg = performance_metrics.cost_with_algorithm(cost_matrix_sample, predictions)
    cost_without_alg = performance_metrics.cost_without_alg(cost_matrix_sample, y_test)

    return cost_with_alg / cost_without_alg


def cv_cslogit_ratio(lambd, fixed_cost, X_train, y_train, X_test,
                     y_test, amount_train, amount_test, cost_matrix_sample):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    lgtr = LogisticRegression(penalty='l2')
    clf = lgtr.fit(X_train, y_train)
    init_theta = np.insert(clf.coef_, 0, values=clf.intercept_)

    logist = cs_logit.cslogit(lambd, fixed_cost)
    logist.fitting(X_train, y_train, amount_train, init_theta)
    cslogitpred = logist.predictcslogit(X_test, threshold=fixed_cost / amount_test)
    cost_with_alg = performance_metrics.cost_with_algorithm(cost_matrix_sample, cslogitpred[1])
    cost_without_alg = performance_metrics.cost_without_alg(cost_matrix_sample, y_test)

    return cost_with_alg / cost_without_alg


def cv_oversample_cslogit_ratio(lambd , fixed_cost, X_train, y_train, X_test,
                                y_test, amount_test, cost_matrix_sample):

    mean_amount = np.mean(X_train[:, -2])
    std_amount = np.std(X_train[:, -2])

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.fit_transform(X_test)

    #sm = SMOTE(sampling_strategy=0.00172, random_state=2290)
    sm = SMOTE(sampling_strategy=0.00400, random_state=2290, k_neighbors= 2)
    X_over_norm, y_over = sm.fit_sample(X_train_norm, y_train)
    amount_over = (X_over_norm[:, -2] * std_amount) + mean_amount

    lgtr = LogisticRegression(penalty='l2')
    clf = lgtr.fit(X_over_norm, y_over)
    init_theta = np.insert(clf.coef_, 0, values=clf.intercept_)

    logist = cs_logit.cslogit(lambd, fixed_cost)
    logist.fitting(X_over_norm, y_over, amount_over, init_theta)
    cslogitpred = logist.predictcslogit(X_test_norm, threshold=fixed_cost / amount_test)
    cost_with_alg = performance_metrics.cost_with_algorithm(cost_matrix_sample, cslogitpred[1])
    cost_without_alg = performance_metrics.cost_without_alg(cost_matrix_sample, y_test)

    return cost_with_alg / cost_without_alg
