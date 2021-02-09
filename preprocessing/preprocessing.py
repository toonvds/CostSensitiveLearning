import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import time

# Set random seed
random.seed(2)
# Suppress pd SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

def convert_categorical_variables(df, labels, categorical_variables):
    # Use weight of evidence encoding: WOE = ln (p(1) / p(0))
    for key in categorical_variables:
        values = np.unique(df[key])
        woe_encoding = np.zeros(len(df))
        for category in values:
            # Get indices of this category
            cat_labels = labels[df[key] == category]
            # Get churn probability of this category
            prob_pos = (cat_labels.sum() / len(cat_labels))
            prob_neg = 1 - prob_pos
            # Minimum value to avoid log(0) / division by 0
            if prob_pos == 0:
                prob_pos = 1e-9
            if prob_neg == 0:
                prob_neg = 1e-9

            # (telco_churn['Churn'] == 'Yes').astype(int)
            woe_encoding[df[key] == category] = np.log(prob_pos / prob_neg)

        df[key + '_WoE'] = woe_encoding

    df = df.drop(categorical_variables, 1)

    return df


# Todo: Standardize function
def standardize(x_train, x_val, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    x_test_scaled = np.ascontiguousarray(x_test_scaled)  # For compatibility with xgboost package

    return x_train_scaled, x_val_scaled, x_test_scaled


def preprocess_credit_card_data(fixed_cost, eda=False):
    """
    Load the Kaggle credit card dataset
    """
    try:
        creditcard = pd.read_csv('data/Kaggle Credit Card Fraud/creditcard.csv')
    except FileNotFoundError:
        creditcard = pd.read_csv('../data/Kaggle Credit Card Fraud/creditcard.csv')

    # Preprocessing
    creditcard = creditcard[creditcard['Amount'] != 0]  # remove transactions with zero Amount
    creditcard = creditcard.drop('Time', 1)  # remove variable Time

    amounts = creditcard['Amount'].values
    labels = creditcard['Class'].values

    # Do not include log(amount)
    # creditcard['LogAmount'] = pd.Series(np.log(creditcard['Amount']))  # log-transformation of Amount

    cols = list(creditcard.columns)  # rearrange some columns - amount final column
    a, b = cols.index('Amount'), cols.index('Class')
    cols[b], cols[a] = cols[a], cols[b]
    creditcard = creditcard[cols]

    # Todo: optionally include next line?
    # creditcard = creditcard.drop('Amount', 1)

    # covariates = np.array(creditcard.drop(['Class'], axis=1))
    covariates = creditcard.drop(['Class'], axis=1)

    # scaler = StandardScaler()
    # covariates_scaled = scaler.fit_transform(covariates)

    # Create cost matrix
    cost_matrix = np.zeros((len(covariates), 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0.0
    cost_matrix[:, 0, 1] = amounts
    cost_matrix[:, 1, 0] = fixed_cost
    cost_matrix[:, 1, 1] = fixed_cost

    # Exploratory data analysis
    if eda:
        # 1. Cost distribution (per class)
        # Kernel density of amounts / logamounts
        # TODO: show fixed cost!
        fig, ax = plt.subplots(1, 2)
        sns.kdeplot(amounts[labels == 0], ax=ax[0])
        sns.kdeplot(amounts[labels == 1], ax=ax[0])
        ax[0].axvline(fixed_cost, 0, 1)
        # sns.kdeplot(amount, ax=ax[0])
        sns.kdeplot(np.log(amounts[labels == 0]), ax=ax[1])
        sns.kdeplot(np.log(amounts[labels == 1]), ax=ax[1])
        ax[1].axvline(np.log(fixed_cost), 0, 1)
        ax[0].set_xscale('log')
        ax[0].set_title('Amount')
        ax[1].set_title('Log(Amount)')
        ax[0].legend(['Negative', 'Positive'])
        ax[1].legend(['Negative', 'Positive'])

        fig.suptitle('Kernel density plots of transaction amounts')
        fig.show()

        # 2. t-SNE for two-dimensional representation (Total sample takes too long, so we use subsample of non-fraud cases)
        start = time.time()
        sample_size = 5000

        neg_indices = np.where(labels == 0)[0]
        sample_neg_indices = np.random.choice(neg_indices, size=sample_size).astype(np.int)
        pos_indices = np.where(labels == 1)[0]

        cov_sc_sample = np.concatenate((covariates[sample_neg_indices], covariates[pos_indices]))
        labels_sample = np.concatenate((labels[sample_neg_indices], labels[pos_indices]))
        amounts_sample = np.concatenate((amounts[sample_neg_indices], amounts[pos_indices]))

        tsne = TSNE(n_components=2, verbose=1, perplexity=9, n_iter=500)
        tsne_results = tsne.fit_transform(cov_sc_sample, labels_sample)

        fig2 = plt.figure()
        df_subset = pd.DataFrame()
        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]
        df_subset['y'] = labels_sample
        df_subset['amounts'] = amounts_sample
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", 2),
            data=df_subset,
            legend="full",
            alpha=0.3,
            s=amounts_sample
        )
        fig2.show()
        end = time.time()
        print('Time elapsed = {}'.format(end - start))

    return covariates, labels, amounts, cost_matrix, []  # No categorical variables


# Todo: complete this dataset
def preprocess_kdd98(eda=False):
    print('TODO')
    kdd98train = pd.read_csv('../data/KDD98 Direct Mailing Donations/cup98LRN.txt')
    kdd98val = pd.read_csv('../data/KDD98 Direct Mailing Donations/cup98VAL.txt')
    kdd98val_labels = pd.read_csv('../data/KDD98 Direct Mailing Donations/valtargt.txt')

    # Attach kdd98val_labels to kdd98val
    kdd98val['CONTROLN'] = kdd98val_labels['CONTROLN']  # Unique ID
    kdd98val['TARGET_B'] = kdd98val_labels['TARGET_B']  # Donated yes/no
    kdd98val['TARGET_D'] = kdd98val_labels['TARGET_D']  # Donated amount

    # Merge train and val
    kdd98 = pd.concat([kdd98train, kdd98val])

    # Todo: preprocessing
    # Encoding categorical variables (weight of evidence encoding)


    # Split into covariates, labels, amounts, cost_matrix


    # return covariates, labels, amounts, cost_matrix, categorical_variables


def preprocess_give_me_some_credit():
    try:
        gmsc_train = pd.read_csv('data/Kaggle Give Me Some Credit/cs-training.csv')
    except FileNotFoundError:
        gmsc_train = pd.read_csv('../data/Kaggle Give Me Some Credit/cs-training.csv')

    # Test set does not contain labels
    # gmsc_test = pd.read_csv('../data/Kaggle Give Me Some Credit/cs-test.csv')

    # Drop ID
    gmsc_train = gmsc_train.drop('Unnamed: 0', 1)

    # EDA: see https://www.kaggle.com/nicholasgah/eda-credit-scoring-top-100-on-leaderboard
    # Preprocessing
    # Follow Bahnsen et al 2014
    # https://github.com/albahnsen/CostSensitiveClassification/blob/master/costcla/datasets/base.py
    gmsc_train = gmsc_train.dropna()
    gmsc_train = gmsc_train.loc[(gmsc_train['MonthlyIncome'] > 0)]
    gmsc_train = gmsc_train.loc[(gmsc_train['DebtRatio'] < 1)]

    # Split into covariates, labels, amounts, cost_matrix
    labels = gmsc_train['SeriousDlqin2yrs'].values.astype(np.int)

    covariates = gmsc_train.drop('SeriousDlqin2yrs', 1)

    # Todo: categorical variables? - WoE encoding

    # Create cost matrix
    income = covariates['MonthlyIncome'].values
    debt = covariates['DebtRatio'].values
    pi_1 = labels.mean()

    cl = v_calculate_cl(k, income, cl_max, debt, int_r, n_term)
    cl_avg = cl.mean()

    n_samples = income.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0.0
    cost_matrix[:, 0, 1] = v_calculate_cost_fn(cl, lgd)
    cost_matrix[:, 1, 0] = v_calculate_cost_fp(cl, int_r, n_term, int_cf, pi_1, lgd, cl_avg)
    cost_matrix[:, 1, 1] = 0.0

    return covariates, labels, cl, cost_matrix, []  # No categorical variables


def preprocess_telco_customer_churn(eda=False):
    try:
        telco_churn = pd.read_csv('data/Kaggle Telco Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except FileNotFoundError:
        telco_churn = pd.read_csv('../data/Kaggle Telco Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Drop customer ID
    telco_churn = telco_churn.drop('customerID', 1)

    # Drop observations with missing values
    telco_churn = telco_churn.replace('', np.nan)
    telco_churn = telco_churn.replace(' ', np.nan)
    telco_churn = telco_churn.dropna()

    # Get labels
    labels = (telco_churn['Churn'] == 'Yes').astype(int)
    telco_churn = telco_churn.drop('Churn', 1)

    # EDA
    if eda:
        for key in telco_churn.keys():
            plt.figure(key)
            plt.title(key)
            if not (key == 'TotalCharges'):
                markers, counts = np.unique(telco_churn[key], return_counts=True)
                plt.hist(markers, weights=counts)
            else:
                sns.kdeplot(telco_churn[key].astype(float))
            plt.show()

    # Convert categorical variables
    # telco_churn['gender'] = pd.factorize(telco_churn['gender'])

    categorical_variables = [
        # 2 categories:
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling',
        # 3 or more categories:        
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    # Create the cost matrix based on Monthly Costs (Petrides and Verbeke, 2020)
    amounts = telco_churn['MonthlyCharges'].to_numpy()

    n_samples = amounts.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0.0
    cost_matrix[:, 0, 1] = 12 * amounts
    cost_matrix[:, 1, 0] = 2 * amounts
    cost_matrix[:, 1, 1] = 0.0

    '''
    # Standardizing
    scaler = StandardScaler()
    covariates_scaled = scaler.fit_transform(covariates)
    '''

    # covariates = telco_churn.to_numpy()
    covariates = telco_churn
    labels = labels.to_numpy()

    return covariates, labels, amounts, cost_matrix, categorical_variables


def preprocess_default_credit_card(eda=False):
    try:
        df = pd.read_excel('data/UCI Default of Credit Card Clients/default of credit card clients.xls', skiprows=[0])
    except FileNotFoundError:
        df = pd.read_excel('../data/UCI Default of Credit Card Clients/default of credit card clients.xls', skiprows=[0])

    # Drop id
    df = df.drop('ID', 1)

    # Get labels
    labels = df['default payment next month'].to_numpy()
    df = df.drop('default payment next month', 1)

    # EDA
    if eda:
        for key in df.keys():
            plt.figure(key)
            plt.title(key)
            markers, counts = np.unique(df[key], return_counts=True)
            plt.hist(markers, weights=counts)
            plt.show()

    # List categorical variables
    categorical_variables = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    # Create the cost matrix based on Bahnsen
    credit_lines = df['LIMIT_BAL'].to_numpy()
    pi_1 = labels.mean()

    n_samples = credit_lines.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0.0
    cost_matrix[:, 0, 1] = v_calculate_cost_fn(credit_lines, lgd)
    cost_matrix[:, 1, 0] = v_calculate_cost_fp(credit_lines, int_r, n_term, int_cf, pi_1, lgd, credit_lines.mean())
    cost_matrix[:, 1, 1] = 0.0

    return df, labels, credit_lines, cost_matrix, categorical_variables


def preprocess_bank_marketing(eda=False):
    try:
        df = pd.read_csv('data/UCI Bank Marketing/bank-full.csv', delimiter=';')
    except FileNotFoundError:
        df = pd.read_csv('../data/UCI Bank Marketing/bank-full.csv', delimiter=';')

    # Get labels
    labels = (df['y'] == 'yes').to_numpy().astype(np.int)
    df = df.drop('y', 1)

    # EDA
    if eda:
        for key in df.keys():
            plt.figure(key)
            plt.title(key)
            markers, counts = np.unique(df[key], return_counts=True)
            plt.hist(markers, weights=counts)
            plt.show()

    # Todo: include duration or not? 0 for label = 0
    # df = df.drop('duration, 1')

    # Convert categorical variables
    categorical_variables = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month',
                             'poutcome']

    # Create the cost matrix based on Bahnsen et al (2014)
    fixed_cost = 1
    interest = df['balance'].to_numpy() * 0.2 * 0.02463333
    expected_interest = np.maximum(interest, fixed_cost)   # Todo: Done for reasonableness condition, BUT SHOULD NOT HOLD on instance-level
    # expected_interest = np.maximum(interest, 0)  # Todo: gives errors later on - division by zero -- FIX!

    n_samples = df.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = fixed_cost
    cost_matrix[:, 0, 1] = fixed_cost
    cost_matrix[:, 1, 0] = expected_interest
    cost_matrix[:, 1, 1] = 0.0

    return df, labels, expected_interest, cost_matrix, categorical_variables


# Todo: make single function!
# Credit scoring - cost_matrix (see Bahnsen et al 2014)
# https://github.com/albahnsen/CostSensitiveClassification/blob/master/costcla/datasets/base.py

# Calculate monthly payment (given credit line cl_i, interest rate int_ and number of terms n_term)
def calculate_a(cl_i, int_, n_term):
    """ Private function """
    return cl_i * ((int_ * (1 + int_) ** n_term) / ((1 + int_) ** n_term - 1))

# Calculate present value (of amount a given interest rate int_ and n_term number of terms)
def calculate_pv(a, int_, n_term):
    """ Private function """
    return a / int_ * (1 - 1 / (1 + int_) ** n_term)

# Calculate credit line Cl
def calculate_cl(k, inc_i, cl_max, debt_i, int_r, n_term):
    """ Private function """
    cl_k = k * inc_i    # k times monthly income
    A = calculate_a(cl_k, int_r, n_term)
    Cl_debt = calculate_pv(inc_i * min(A / inc_i, 1 - debt_i), int_r, n_term)
    return min(cl_k, cl_max, Cl_debt)

# Calculate cost of FN
def calculate_cost_fn(cl_i, lgd):
    return cl_i * lgd

# Calculate cost of FP
def calculate_cost_fp(cl_i, int_r, n_term, int_cf, pi_1, lgd, cl_avg):
    # Lost profit
    a = calculate_a(cl_i, int_r, n_term)
    pv = calculate_pv(a, int_cf, n_term)
    r = pv - cl_i

    # Profit of average customer
    r_avg = calculate_pv(calculate_a(cl_avg, int_r, n_term), int_cf, n_term) - cl_avg
    cost_fp = r - (1 - pi_1) * r_avg + pi_1 * calculate_cost_fn(cl_avg, lgd)
    return max(0, cost_fp)

# Apply functions to arrays
v_calculate_cost_fp = np.vectorize(calculate_cost_fp)
v_calculate_cost_fn = np.vectorize(calculate_cost_fn)
v_calculate_cl = np.vectorize(calculate_cl)

# Set variables/assumptions
k = 3
int_r = 0.0479 / 12
n_term = 24
int_cf = 0.0294 / 12
lgd = .75
cl_max = 25000
