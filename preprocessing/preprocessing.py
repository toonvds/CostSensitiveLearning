import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import woe
import time

# Set random seed
random.seed(2)
# Suppress pd SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


def convert_categorical_variables(x_train, y_train, x_val, x_test, categorical_variables):
    # Use weight of evidence encoding: WOE = ln (p(1) / p(0))

    woe_encoder = woe.WOEEncoder(verbose=1, cols=categorical_variables)
    woe_encoder.fit(x_train, y_train)
    x_train = woe_encoder.transform(x_train)
    x_val = woe_encoder.transform(x_val)
    x_test = woe_encoder.transform(x_test)

    # for key in categorical_variables:
    #     values = pd.unique(df[key])
    #     # values = np.unique(df[key])
    #     woe_encoding = np.zeros(len(df))
    #     for category in values:
    #         # Get indices of this category
    #         cat_labels = labels[df[key] == category]
    #         # Get class 1/0 probability of this category
    #         prob_pos = (cat_labels.sum() / len(cat_labels))
    #         prob_neg = 1 - prob_pos
    #
    #         # Minimum value to avoid log(0) / division by 0
    #         if prob_pos == 0:
    #             prob_pos = 1e-9
    #         if prob_neg == 0:
    #             prob_neg = 1e-9
    #
    #         woe_encoding[df[key] == category] = np.log(prob_pos / prob_neg)
    #
    #     df[key + '_WoE'] = woe_encoding
    #
    # df = df.drop(categorical_variables, 1)

    return x_train, x_val, x_test


def standardize(x_train, x_val, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # For compatibility with xgboost package: make contiguous arrays
    x_train_scaled = np.ascontiguousarray(x_train_scaled)
    x_val_scaled = np.ascontiguousarray(x_val_scaled)
    x_test_scaled = np.ascontiguousarray(x_test_scaled)

    return x_train_scaled, x_val_scaled, x_test_scaled


def handle_missing_data(df_train, df_val, df_test, categorical_variables):

    for key in df_train.keys():
        # If variable has > 90% missing values: delete
        if df_train[key].isna().mean() > 0.9:
            df_train = df_train.drop(key, 1)
            df_val = df_val.drop(key, 1)
            df_test = df_test.drop(key, 1)

            if key in categorical_variables:
                categorical_variables.remove(key)
            continue

        # Handle other missing data:
        #   Categorical variables: additional category '-1'
        if key in categorical_variables:
            df_train[key] = df_train[key].fillna('-1')
            df_val[key] = df_val[key].fillna('-1')
            df_test[key] = df_test[key].fillna('-1')
        #   Continuous variables: median imputation
        else:
            median = df_train[key].median()
            df_train[key] = df_train[key].fillna(median)
            df_val[key] = df_val[key].fillna(median)
            df_test[key] = df_test[key].fillna(median)

    assert df_train.isna().sum().sum() == 0 and df_val.isna().sum().sum() == 0 and df_test.isna().sum().sum() == 0

    return df_train, df_val, df_test, categorical_variables


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


def preprocess_kdd98(eda=False):
    try:
        kdd98train = pd.read_csv('data/KDD98 Direct Mailing Donations/cup98LRN.txt', low_memory=False)
        kdd98val = pd.read_csv('data/KDD98 Direct Mailing Donations/cup98VAL.txt', low_memory=False)
        kdd98val_labels = pd.read_csv('data/KDD98 Direct Mailing Donations/valtargt.txt', low_memory=False)
    except FileNotFoundError:
        kdd98train = pd.read_csv('../data/KDD98 Direct Mailing Donations/cup98LRN.txt', low_memory=False)
        kdd98val = pd.read_csv('../data/KDD98 Direct Mailing Donations/cup98VAL.txt', low_memory=False)
        kdd98val_labels = pd.read_csv('../data/KDD98 Direct Mailing Donations/valtargt.txt', low_memory=False)

    # Attach kdd98val_labels to kdd98val
    kdd98val['CONTROLN'] = kdd98val_labels['CONTROLN']  # Unique ID
    kdd98val['TARGET_B'] = kdd98val_labels['TARGET_B']  # Donated yes/no
    kdd98val['TARGET_D'] = kdd98val_labels['TARGET_D']  # Donated amount

    # Merge train and val
    kdd98 = pd.concat([kdd98train, kdd98val])

    # Keep only subset of attributes - following Petrides and Verbeke (2020)
    selected_attributes = ['MAILCODE', 'NOEXCH', 'AGE', 'HOMEOWNR', 'NUMCHLD', 'INCOME', 'GENDER', 'WEALTH1',
                           'COLLECT1', 'CARDPROM', 'MAXADATE', 'CARDGIFT', 'MINRAMNT', 'MINRDATE', 'MAXRAMNT',
                           'MAXRDATE', 'LASTGIFT', 'LASTDATE', 'FISTDATE', 'NEXTDATE', 'TIMELAG', 'AVGGIFT',
                           'TARGET_D', 'TARGET_B']

    kdd98 = kdd98[selected_attributes]

    # Get amounts and labels
    amounts = kdd98['TARGET_D'].to_numpy()
    labels = kdd98['TARGET_B'].to_numpy()
    kdd98 = kdd98.drop(['TARGET_D', 'TARGET_B'], 1)

    # List categorical variables
    categorical_variables = ['MAILCODE', 'NOEXCH', 'HOMEOWNR', 'GENDER', 'COLLECT1', 'MAXADATE', 'MINRDATE', 'MAXRDATE',
                             'LASTDATE', 'FISTDATE', 'NEXTDATE']

    # Following Petrides and Verbeke, we use a linear regression to estimate amounts for people who did not donate
    # Rough prediction: leave out categorical variables + variables with any missing values
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    x_temp = kdd98.drop(categorical_variables, 1)
    x_temp = x_temp.dropna(axis=1)
    lin_reg.fit(X=x_temp[labels == 1], y=amounts[labels == 1])
    est_amounts = lin_reg.predict(X=x_temp[labels == 0])
    amounts[labels == 0] = est_amounts

    # Cost matrix   Todo: Follow Petrides and Verbeke 2020?
    n_samples = amounts.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0                            # Instead of 0.68  (no cost for contacting)
    cost_matrix[:, 0, 1] = amounts  # amounts[labels == 1].mean()  # Instead of 14.45 (no cost for contacting)
    cost_matrix[:, 1, 0] = 0.68
    cost_matrix[:, 1, 1] = 0.68

    return kdd98, labels, amounts, cost_matrix, categorical_variables


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

    # Preprocessing follows Bahnsen et al 2014
    # https://github.com/albahnsen/CostSensitiveClassification/blob/master/costcla/datasets/base.py
    gmsc_train = gmsc_train.dropna()
    gmsc_train = gmsc_train.loc[(gmsc_train['MonthlyIncome'] > 0)]
    gmsc_train = gmsc_train.loc[(gmsc_train['DebtRatio'] < 1)]

    # Split into covariates, labels, amounts, cost_matrix
    labels = gmsc_train['SeriousDlqin2yrs'].values.astype(np.int)
    covariates = gmsc_train.drop('SeriousDlqin2yrs', 1)

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
    cost_matrix[:, 1, 1] = 0

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

    # Create the cost matrix based on Bahnsen  # Todo: ref
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

    # See S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing.
    # Decision Support Systems, Elsevier, 62:22-31, June 2014

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
    df = df.drop('duration', 1)

    # Convert categorical variables
    categorical_variables = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month',
                             'poutcome']

    # Create the cost matrix - following Bahnsen et al (2015)
    fixed_cost = 1
    interest = df['balance'].to_numpy() * 0.2 * 0.02463333
    expected_interest = np.maximum(interest, fixed_cost)   # Todo: Done for reasonableness condition, BUT SHOULD NOT HOLD on instance-level
    # expected_interest = np.maximum(interest, 0)  # Todo: gives errors later on - division by zero -- FIX!

    n_samples = df.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0.0
    cost_matrix[:, 0, 1] = expected_interest
    cost_matrix[:, 1, 0] = fixed_cost
    cost_matrix[:, 1, 1] = fixed_cost

    return df, labels, expected_interest, cost_matrix, categorical_variables


def preprocess_vub_credit_scoring(eda=False):
    try:
        df = pd.read_csv('data/VUB Credit Scoring/anonymized_dataset.csv', delimiter=';')
    except FileNotFoundError:
        df = pd.read_csv('../data/VUB Credit Scoring/anonymized_dataset.csv', delimiter=';')

    # First instance has V1 as Days_late: delete
    df = df.drop(0)

    # Drop ID + original test set partitioning
    df = df.drop('ID', 1)
    df = df.drop(['Test_set1', 'Test_set2', 'Test_set3'], 1)
    # Todo: Differentiate between business channels?
    # df = df[df['Business_channel'] == 3]

    # Get labels
    labels = (df['Default_45']).to_numpy().astype(np.int)
    df = df.drop(['Default_45', 'Days_late'], 1)

    # Todo: Set FICO_Score to average for missing values (see Petrides et al. 2020)
    df['FICO_Score'][df['FICO_Score'].isna()] = 0
    # df = df.drop('FICO_Score', 1)

    # EDA
    if eda:
        for key in df.keys():
            plt.figure(key)
            plt.title(key)
            markers, counts = np.unique(df[key], return_counts=True)
            plt.hist(markers, weights=counts)
            plt.show()

    # List categorical variables
    categorical_variables = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'Has_FICO', 'Business_channel']

    # Create the cost matrix based on Petrides et al. 2020
    amounts = df['Loan_amount'].to_numpy()

    # Todo: Allow negative costs?
    # Scale amounts such that they are all positive
    amounts = amounts - amounts.min() + 1e-9

    df = df.drop(['Expected_loss', 'Expected_profit'], 1)

    # Alternative: Bahnsen strategy
    pi_1 = labels.mean()

    n_samples = amounts.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))     # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0.0
    cost_matrix[:, 0, 1] = v_calculate_cost_fn(amounts, lgd)
    cost_matrix[:, 1, 0] = v_calculate_cost_fp(amounts, int_r, n_term, int_cf, pi_1, lgd, amounts.mean())
    cost_matrix[:, 1, 1] = 0.0

    return df, labels, amounts, cost_matrix, categorical_variables


def preprocess_tv_subscription_churn(eda=False):
    try:
        df = pd.read_csv('data/TV Subscription Churn/churn_tv_subscriptions.csv.gz', delimiter=',', compression='gzip')
    except FileNotFoundError:
        df = pd.read_csv('../data/TV Subscription Churn/churn_tv_subscriptions.csv.gz', delimiter=',',
                         compression='gzip')

    # Drop ID
    df = df.drop('id', 1)

    # Get labels
    labels = df['target'].to_numpy()
    df = df.drop('target', 1)

    if eda:
        for key in df.keys():
            plt.figure(key)
            plt.title(key)
            markers, counts = np.unique(df[key], return_counts=True)
            plt.hist(markers, weights=counts)
            plt.show()

    # List categorical variables (none here)
    categorical_variables = []

    # Extract the cost matrix - based on Bahnsen et al. 2015
    n_samples = labels.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))  # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = df['C_TN']
    cost_matrix[:, 0, 1] = df['C_FN']
    cost_matrix[:, 1, 0] = df['C_FP']
    cost_matrix[:, 1, 1] = df['C_TP']

    clv = df['C_FN']

    df = df.drop(['C_TN', 'C_FN', 'C_FP', 'C_TP'], 1)

    return df, labels, clv, cost_matrix, categorical_variables


def preprocess_kaggle_ieee_fraud(eda=False, subsample=1):
    try:
        train_identity = pd.read_csv('data/IEEE Fraud Detection/train_identity.csv')
        train_transaction = pd.read_csv('data/IEEE Fraud Detection/train_transaction.csv')
    except FileNotFoundError:
        train_identity = pd.read_csv('../data/IEEE Fraud Detection/train_identity.csv')
        train_transaction = pd.read_csv('../data/IEEE Fraud Detection/train_transaction.csv')
        # test_identity = pd.read_csv('../data/IEEE Fraud Detection/test_identity.csv')
        # test_transaction = pd.read_csv('../data/IEEE Fraud Detection/test_transaction.csv')

    # Only use training data as test data does not include labels
    df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    df = df.drop('TransactionID', 1)
    df = df.drop('TransactionDT', 1)

    # Take subset for computational speedup
    if subsample < 1:
        print('Taking only a subset of the data')
        np.random.seed(42)
        drop_indices = np.random.choice(df.index, int((1-subsample)*len(df)), replace=False)
        df = df.drop(drop_indices)

    if eda:
        for key in df.keys():
            print(key)
            print(pd.unique(df[key]))

    # Categorical variables
    cat_var_transaction = ['ProductCD',
                           'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                           'addr1', 'addr2',
                           'P_emaildomain', 'R_emaildomain',
                           'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
    cat_var_identity = ['DeviceType', 'DeviceInfo',
                        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21',
                        'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31',
                        'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']
    categorical_variables = cat_var_transaction + cat_var_identity

    # # Replace NaN with new value "-1" (used nowhere else)
    # df[categorical_variables] = df[categorical_variables].fillna(value=-1)
    #
    # # For non-categorical variables, drop if more than 10% missing values  # Todo: make faster?/more elegant?
    # non_cat_var = [x for x in df.keys().to_list() if x not in categorical_variables]
    # non_cat_var_remaining = []
    # for key in non_cat_var:
    #     if df[key].isna().mean() > 0.01:
    #         df = df.drop(key, 1)
    #     else:
    #         non_cat_var_remaining.append(key)
    # # Drop observations with remaining missing values for non-categorical variables
    # df = df.drop(df[df[non_cat_var_remaining].isna().sum(axis=1) > 0].index)

    # Get labels and amounts
    labels = df['isFraud'].to_numpy()
    df = df.drop('isFraud', 1)
    amounts = df['TransactionAmt'].to_numpy()

    # Make cost matrix
    fixed_cost = 10

    n_samples = labels.shape[0]
    cost_matrix = np.zeros((n_samples, 2, 2))  # cost_matrix [[TN, FN], [FP, TP]]
    cost_matrix[:, 0, 0] = 0.0
    cost_matrix[:, 0, 1] = amounts
    cost_matrix[:, 1, 0] = fixed_cost
    cost_matrix[:, 1, 1] = fixed_cost

    return df, labels, amounts, cost_matrix, categorical_variables


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
