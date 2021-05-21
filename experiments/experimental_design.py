import numpy as np
import pandas as pd

def experimental_design(clas, amount):

    amounts = clas*amount
    fraud_amounts = amounts[amounts != 0]
    value1 = np.quantile(fraud_amounts, 0.33)
    value2 = np.quantile(fraud_amounts, 0.66)

    amounts = pd.Series(amounts)
    # Make sure value 1 != value 2 for pd.cut
    if value1 == value2:
        # value2 += 1e-9
        prepr = pd.cut(amounts, bins=[-1, 0, value1, np.max(amounts)], right=True, labels=[0, 1, 2])
    else:
        prepr = pd.cut(amounts, bins=[-1, 0, value1, value2, np.max(amounts)], right=True, labels=[0, 1, 2, 3])
    '''
    if not(len(np.unique(amounts)) == 2):
        prepr = pd.cut(amounts, bins=[-1, 0, value1, value2, np.max(amounts)], right=True, labels=[0, 1, 2, 3])
    else:  # Class-dependent costs, assign each instance 0, 1, 2 or 3 randomly
        prepr = np.random.choice([0, 1, 2, 3], len(amounts))
    '''
    prepr = np.array(prepr)

    return prepr
