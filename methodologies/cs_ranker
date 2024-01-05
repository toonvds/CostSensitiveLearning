import xgboost as xgb
from scipy.special import expit
from scipy.stats import lognorm
import numpy as np
import pandas as pd
import ctypes
from pyltr import models, util, metrics
# from pyltr.util.sort import get_sorted_y_positions
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import label_ranking_average_precision_score

np.random.seed(seed=42)


class CSRanker:
    def __init__(self, obj, lambda1=0, lambda2=0, min_child_weight=0, ndcg_options={}):
        self.obj = obj
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # alpha is l1, lambda is l2
        # min_child_weight=0 needed for some ranking objectives
        params = {'random_state': 42, 'tree_method': 'exact', 'verbosity': 0, 'reg_alpha': lambda1,
                  'reg_lambda': lambda2, 'min_child_weight': min_child_weight, 'disable_default_eval_metric': 1,
                  'n_jobs': 1}

        # Select objective function (if built-in)
        if obj == 'pairwise':
            params['objective'] = 'rank:pairwise'
        elif obj == 'map':
            params['objective'] = 'rank:map'
        elif obj == 'ndcg':
            params['objective'] = 'rank:ndcg'

        self.params = params
        self.ndcg_options = ndcg_options
        if not ndcg_options:    # Use defaults if no options specified
            self.ndcg_options['gain_type'] = 'identity'
            self.ndcg_options['discount_type'] = 'log2'
            self.ndcg_options['k'] = 'N'

    def fit(self, x_train, y_train, x_val, y_val, cost_matrix_train=None, cost_matrix_val=None):

        # Use NDCG for validation checks
        def metric_val(raw_scores, y_true):
            y_pred = expit(raw_scores)
            y = y_true.get_label()

            if self.ndcg_options['k'] == 'N':
                k_val = len(y)
            else:
                k_val = min(len(y), self.ndcg_options['k'])

            # Calculate dcg
            rankings = np.argsort(y_pred)[::-1]
            sorted_relevance = y[rankings]

            # Apply gain
            # (gain_type 'identity' just continues)
            if self.ndcg_options['gain_type'] == 'exp2':
                sorted_relevance = 2**sorted_relevance - 1

            # Apply discount
            if self.ndcg_options['discount_type'] == 'log2':
                indices = np.cumsum(np.ones(len(y)))
                discount = 1 / np.log2(indices + 1)
            if self.ndcg_options['discount_type'] == 'log10':
                indices = np.cumsum(np.ones(len(y)))
                discount = 1 / np.log10(indices + 1)
            elif self.ndcg_options['discount_type'] == 'linear':
                indices = np.cumsum(np.ones(len(y)))
                discount = np.maximum(1 - (indices - 1) / k_val, 0)
            elif self.ndcg_options['discount_type'] == 'step':
                discount = np.concatenate((np.ones(k_val), np.zeros(len(y) - k_val)))
            elif self.ndcg_options['discount_type'] == 'lognormal':
                indices = np.cumsum(np.ones(len(y)))
                discount = 1 - lognorm.cdf(x=indices, s=1, loc=0, scale=100)
                # lognorm.cdf() parameters correspond to: mean=log(scale), sigma=s

            dcg = sorted_relevance * discount

            ## Calculate ideal
            ideal_relevance = np.sort(y)[::-1]

            ideal = ideal_relevance * discount

            # Todo: only top k instances!!
            return 'NDCG', dcg[0:k_val-1].sum() / ideal[0:k_val-1].sum()

        if self.obj == 'pairwise' or self.obj == 'map' or self.obj == 'ndcg':
            # Include query id as first "feature"
            dtrain = xgb.DMatrix(data=np.hstack((np.ones((len(y_train), 1)), x_train)))
            dtrain.set_label(label=y_train)
            dval = xgb.DMatrix(data=np.hstack((np.ones((len(y_val), 1)), x_val)))
            dval.set_label(label=y_val)

            xgboost = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=500, early_stopping_rounds=50,
                                feval=metric_val, maximize=True, evals=[(dval, 'eval')], verbose_eval=0)

            return xgboost

        elif self.obj == 'ndcg_custom':
            # Include group as first "feature"
            dtrain = xgb.DMatrix(data=np.hstack((np.ones((len(y_train), 1)), x_train)))
            dtrain.set_label(label=y_train)
            dval = xgb.DMatrix(data=np.hstack((np.ones((len(y_val), 1)), x_val)))
            dval.set_label(label=y_val)

            # Use custom NDCG metric
            # Due to quadratic complexity, we will take a sample if N is too large (>N_max)
            # (Only for training purposes)
            N_max = 20000
            if self.ndcg_options['k'] == 'N':
                print('N = {}'.format(len(y_train)))
                if len(y_train) > N_max:
                    print('N too large. Using subsample of ' + str(N_max) + ' during each training iteration.')
                    # Adjust discount function
                    metric = NDCG(k=len(y_train),
                                  gain_type=self.ndcg_options['gain_type'],
                                  discount_type=self.ndcg_options['discount_type'],
                                  N=len(y_train),
                                  subset=N_max / len(y_train)   # Take only a subset (in [0, 1])
                                  )
                else:
                    print('Using entire training set.')
                    metric = NDCG(k=len(y_train),
                                  gain_type=self.ndcg_options['gain_type'],
                                  discount_type=self.ndcg_options['discount_type'],
                                  N=len(y_train)
                                  )
            else:
                metric = NDCG(k=self.ndcg_options['k'],
                              gain_type=self.ndcg_options['gain_type'],
                              discount_type=self.ndcg_options['discount_type'],
                              N=len(y_train)
                              )

            # Custom objective not supported for ranking, but we do not need group IDs
            # Modified from PyLTR, see:
            # https://github.com/jma127/pyltr/blob/c419840432a3c33bc29eece06a679888f1624f46/pyltr/models/lambdamart.py#L264
            def risk_train(raw_scores, y_true):
                # Due to quadratic complexity, we will take a sample if N is too large (>N_max)
                # This will be stratified based on label (not on amount) if label is binary
                # (Only for training purposes)
                N_original = len(raw_scores)
                if N_original > N_max:
                    y = y_true.get_label()

                    # Binary label, stratify:
                    if len(np.unique(y)) == 2:
                        instances_pos = np.argwhere(y == 1).flatten()
                        instances_neg = np.argwhere(y == 0).flatten()

                        # Take subset
                        subset_pos = np.random.choice(instances_pos, size=round(N_max*y.mean()), replace=False)
                        subset_neg = np.random.choice(instances_neg, size=round(N_max*(1 - y.mean())), replace=False)
                        subset = np.concatenate([subset_pos, subset_neg])
                    # Else, take random subset (not stratified):
                    else:
                        ## Stratify:
                        # value1 = np.quantile(y, 0.33)
                        # value2 = np.quantile(y, 0.66)
                        #
                        # bin_labels = pd.cut(y, bins=[np.min(y), value1, value2, np.max(y)], right=True,
                        #                labels=[1, 2, 3])
                        #
                        # # Take subsets
                        # instances1 = np.argwhere(bin_labels == 1).flatten()
                        # instances2 = np.argwhere(bin_labels == 2).flatten()
                        # instances3 = np.argwhere(bin_labels == 3).flatten()
                        #
                        # subset1 = np.random.choice(instances1, size=round(N_max / 3), replace=False)
                        # subset2 = np.random.choice(instances2, size=round(N_max / 3), replace=False)
                        # subset3 = np.random.choice(instances3, size=N_max - 2 * round(N_max / 3), replace=False)
                        #
                        # subset = np.concatenate([subset1, subset2, subset3])

                        instances = np.cumsum(np.ones(len(y)), dtype=np.int) - 1
                        subset = np.random.choice(instances, size=N_max, replace=False)

                    raw_scores = raw_scores[subset]
                    y = y[subset]
                else:
                    y = y_true.get_label()

                y_pred = expit(raw_scores)

                ns = y.shape[0]

                # Get actual ranking:
                positions = util.sort.get_sorted_y_positions(y, y_pred, check=False)
                actual = y[positions]

                swap_deltas = metric.calc_swap_deltas(qid=1.0, targets=actual)  # |Delta NDCG|
                max_k = metric.max_k()
                if max_k is None or ns < max_k:
                    max_k = ns

                lambdas = np.zeros(ns)
                deltas = np.zeros(ns)

                # Loop over first k instances
                for i in range(max_k):
                    # Check all remaining pairs
                    for j in range(i + 1, ns):
                        if actual[i] == actual[j]:
                            continue

                        delta_metric = swap_deltas[i, j]
                        if delta_metric == 0.0:
                            continue

                        a, b = positions[i], positions[j]
                        # invariant: y_pred[a] >= y_pred[b]

                        if actual[i] < actual[j]:
                            assert delta_metric > 0.0
                            logistic = expit(y_pred[a] - y_pred[b])
                            l = logistic * delta_metric
                            lambdas[a] -= l     # Should be -l!
                            lambdas[b] += l
                        else:
                            assert delta_metric < 0.0
                            logistic = expit(y_pred[b] - y_pred[a])
                            l = logistic * -delta_metric
                            lambdas[a] += l
                            lambdas[b] -= l

                        gradient = (1 - logistic) * l
                        deltas[a] += gradient
                        deltas[b] += gradient

                # print('Gradient and hessian:')
                # print(lambdas.mean())
                # print(deltas.mean())

                # If we subsampled, get full sample of gradients (with 0 for unused instances):
                if N_original > N_max:
                    lambdas_full = np.zeros(N_original)
                    deltas_full = np.zeros(N_original)
                    lambdas_full[subset] = lambdas
                    deltas_full[subset] = deltas

                    lambdas = lambdas_full
                    deltas = deltas_full

                # return gradient and hessian
                return -lambdas, deltas  # Pyltr's gradient is other way around, so added -

            # def risk_val(raw_scores, y_true):
            #
            #     # Return risk (not grad/hess)
            #
            #     return 'Risk', risk

            xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=risk_train, feval=metric_val,  maximize=True,
                                num_boost_round=500, early_stopping_rounds=50, evals=[(dval, 'eval')],
                                verbose_eval=True)

            return xgboost

        elif self.obj == 'pyltr':
            # NDCG
            # metric = metrics.NDCG(k=10)
            metric = metrics.NDCG(k=len(y_train))

            # Only needed if you want to perform validation (early stopping & trimming)
            monitor = models.monitors.ValidationMonitor(
                x_val, y_val, np.ones(len(y_val)), metric=metric, stop_after=20)

            model = models.LambdaMART(
                metric=metric,
                n_estimators=500,
                learning_rate=1,
                subsample=1.0,
                query_subsample=1,
                max_features=1,
                max_leaf_nodes=5,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                verbose=1,
                random_state=42,
                warm_start=False
            )

            model.fit(x_train, y_train, np.ones(len(y_train)), monitor=monitor)

            return model


# import numpy as np
from pyltr.metrics import gains, Metric
from six import moves

_EPS = np.finfo(np.float64).eps
range = moves.range

# Define custom metric:
# This is done by adapting pyltr's NDCG metric
# See https://github.com/jma127/pyltr/blob/master/pyltr/metrics/dcg.py
# We do not need the qid
class DCG(Metric):
    # Gain type: either 'exp2' or 'identity'
    def __init__(self, N, k=10, gain_type='exp2', discount_type='log', subset=1):
        # super(DCG, self).__init__()
        self.k = k
        self.gain_type = gain_type
        self._gain_fn = gains.get_gain_fn(gain_type)

        self.discount_type = discount_type
        if discount_type == 'log2':
            # self._discounts = self._make_discounts(n=k)
            indices = np.cumsum(np.ones(N))
            self._discounts = 1 / np.log2(indices + 1)
            self._discounts[k:] = 0
        if discount_type == 'log10':
            # self._discounts = self._make_discounts(n=k)
            indices = np.cumsum(np.ones(N))
            self._discounts = 1 / np.log10(indices + 1)
            self._discounts[k:] = 0
        elif discount_type == 'linear':
            indices = np.cumsum(np.ones(N))
            self._discounts = np.maximum(1 - (indices - 1) / k, 0)
            self._discounts[k:] = 0
        elif discount_type == 'step':
            self._discounts = np.concatenate((np.ones(k), np.zeros(N - k)))
            self._discounts[k:] = 0
        elif discount_type == 'lognormal':
            indices = np.cumsum(np.ones(N))
            self._discounts = 1 - lognorm.cdf(x=indices, s=1, loc=0, scale=100)
            self._discounts[k:] = 0
        if subset < 1:
            # Take discounts over entire N, and take a subset:
            N_subset = round(N * subset)
            indices_subset = np.array(np.cumsum(np.ones(N_subset)) / subset, dtype=np.int)
            self._discounts = self._discounts[indices_subset - 1]


    def evaluate(self, qid, targets):
        return sum(self._gain_fn(t) * self._get_discount(i)
                   for i, t in enumerate(targets) if i < self.k)

    def calc_swap_deltas(self, qid, targets, coeff=1.0):
        n_targets = len(targets)
        deltas = np.zeros((n_targets, n_targets))

        for i in range(min(n_targets, self.k)):
            for j in range(i + 1, n_targets):
                deltas[i, j] = coeff * \
                               (self._gain_fn(targets[i]) - self._gain_fn(targets[j])) * \
                               (self._get_discount(j) - self._get_discount(i))

        return deltas

    def max_k(self):
        return self.k

    def calc_random_ev(self, qid, targets):
        total_gains = sum(self._gain_fn(t) for t in targets)
        total_discounts = sum(self._get_discount(i)
                              for i in range(min(self.k, len(targets))))
        return total_gains * total_discounts / len(targets)

    # Discount metric:
    @classmethod
    def _make_discounts(self, n):
        return np.array([1.0 / np.log2(i + 2.0) for i in range(n)])

    def _get_discount(self, i):
        # Get discount for first i elements (or i-1 if starting from 0):
        if i >= self.k:
            return 0.0
        while i >= len(self._discounts):
            self._grow_discounts()

        return self._discounts[i]

    def _grow_discounts(self):
        if self.discount_type == 'log2':
            self._discounts = self._make_discounts(len(self._discounts) * 2)
        elif self.discount_type == 'log10':
            print('_grow_discounts not implemented for discount_type log10')
        elif self.discount_type == 'linear':
            print('_grow_discounts not implemented for discount_type linear')
        elif self.discount_type == 'step':
            print('_grow_discounts not implemented for discount_type step')
        elif self.discount_type == 'lognormal':
            print('_grow_discounts not implemented for discount_type lognormal')


class NDCG(Metric):
    def __init__(self, N, k=10, gain_type='exp2', discount_type='log2', subset=1):
        super(NDCG, self).__init__()
        self.k = k
        self.gain_type = gain_type
        self._dcg = DCG(N=N, k=k, gain_type=gain_type, discount_type=discount_type, subset=subset)
        self._ideals = {}

    def evaluate(self, qid, targets):
        return (self._dcg.evaluate(qid, targets) /
                max(_EPS, self._get_ideal(qid, targets)))

    def calc_swap_deltas(self, qid, targets):
        ideal = self._get_ideal(qid, targets)
        if ideal < _EPS:
            return np.zeros((len(targets), len(targets)))
        return self._dcg.calc_swap_deltas(
            qid, targets, coeff=1.0 / ideal)

    def max_k(self):
        return self.k

    def calc_random_ev(self, qid, targets):
        return (self._dcg.calc_random_ev(qid, targets) /
                max(_EPS, self._get_ideal(qid, targets)))

    def _get_ideal(self, qid, targets):
        ideal = self._ideals.get(qid)
        if ideal is not None:
            return ideal
        sorted_targets = np.sort(targets)[::-1]
        ideal = self._dcg.evaluate(qid, sorted_targets)
        self._ideals[qid] = ideal
        return ideal
