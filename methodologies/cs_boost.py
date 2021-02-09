import xgboost as xgb
from scipy.special import expit
import numpy as np

# Todo: make wrapper?
class CSBoost:
    def __init__(self, obj, lambda1=0, lambda2=0):
        self.obj = obj

        # alpha is l1, lambda is l2
        params = {'random_state': 42, 'tree_method': 'exact', 'verbosity': 0, 'reg_alpha': lambda1,
                  'reg_lambda': lambda2}
        if obj == 'ce' or obj == 'weightedce':
            params['objective'] = 'binary:logistic'
        elif obj == 'aec':
            params['disable_default_eval_metric'] = True

        self.params = params

    def fit(self, x_train, y_train, x_val, y_val, cost_matrix_train=None, cost_matrix_val=None):
        if self.obj == 'ce':
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dval = xgb.DMatrix(x_val, label=y_val)

            xgboost = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=500, early_stopping_rounds=50,
                            evals=[(dval, 'eval')], verbose_eval=False)

        elif self.obj == 'weightedce':
            misclass_costs = np.zeros(len(y_train))
            misclass_costs[y_train == 0] = cost_matrix_train[:, 1, 0][y_train == 0]
            misclass_costs[y_train == 1] = cost_matrix_train[:, 0, 1][y_train == 1]

            misclass_costs_val = np.zeros(len(y_val))
            misclass_costs_val[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
            misclass_costs_val[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

            dtrain = xgb.DMatrix(x_train, label=y_train, weight=misclass_costs)
            dval = xgb.DMatrix(x_val, label=y_val, weight=misclass_costs_val)

            xgboost = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=500, early_stopping_rounds=50,
                                evals=[(dval, 'eval')], verbose_eval=False)

        elif self.obj == 'aec':
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dval = xgb.DMatrix(x_val, label=y_val)

            # Do constant computations here to avoid DMatrix error
            # diff_costs_train = fixed_cost - y_train * amounts_train

            train_constant = (y_train * (cost_matrix_train[:, 1, 1] - cost_matrix_train[:, 0, 1])
                             + (1 - y_train) * (cost_matrix_train[:, 1, 0] - cost_matrix_train[:, 0, 0]))

            def aec_train(raw_scores, y_true):
                scores = expit(raw_scores)

                # Average expected cost:
                # ec = np.multiply(np.multiply(y_true, (1 - scores)), amounts_train) + np.multiply(scores, fixed_cost)
                # ec = y_true * (
                #     scores * cost_matrix_train[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1]) \
                #     + (1 - y_true) * (
                #     scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                # Gradient
                # Use diff_costs_train instead of (fixed_cost - y_true*amounts_train)
                # grad = scores * (1 - scores) * diff_costs_train
                grad = scores * (1 - scores) * train_constant

                # Hessian
                hess = np.abs((1 - 2 * scores) * grad)
                # hess = scores * (1 - scores) * (1 - 2 * scores) * train_constant

                return grad, hess

            def aec_val(raw_scores, y_true):
                scores = expit(raw_scores)

                # Return AEC (not grad/hess)
                # ec = (1 - scores) * y_val * amounts_val + scores * fixed_cost
                # ec = y_true * (
                #     scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1]) \
                #     + (1 - y_true) * (
                #     scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                # Avoid computations with y_true (DMatrix)
                # Todo: what happens if y_true is a vector?
                if y_true:
                    ec = scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1]
                else:
                    ec = scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0]

                aec = ec.mean()

                return 'AEC', aec

            xgboost = xgb.train(params=self.params, dtrain=dtrain, obj=aec_train, feval=aec_val, num_boost_round=500,
                                early_stopping_rounds=50, evals=[(dval, 'eval')], verbose_eval=False)

        # print('\tBest number of trees = %i' % xgboost.best_ntree_limit)

        return xgboost

    def tune(self, l1, lambda1_list, l2, lambda2_list, x_train, y_train, cost_matrix_train, x_val, y_val,
             cost_matrix_val):
        if l1:
            self.params['reg_lambda'] = 0
            losses_list = []
            for lambda1 in lambda1_list:
                xgboost = CSBoost(obj=self.obj, lambda1=lambda1)
                xgboost = xgboost.fit(x_train, y_train, x_val, y_val, cost_matrix_train, cost_matrix_val)
                scores = xgboost.inplace_predict(x_val)

                # Evaluate loss (without regularization term!)
                if self.obj == 'ce':
                    eps = 1e-9  # small value to avoid log(0)
                    ce = - (y_val * np.log(scores + eps) + (1 - y_val) * np.log(1 - scores + eps))
                    val_loss = ce.mean()
                elif self.obj == 'weightedce':
                    eps = 1e-9  # small value to avoid log(0)
                    ce = - (y_val * np.log(scores + eps) + (1 - y_val) * np.log(1 - scores + eps))

                    cost_misclass = np.zeros(len(y_val))
                    cost_misclass[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                    cost_misclass[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                    weighted_ce = cost_misclass * ce
                    val_loss = weighted_ce.mean()
                elif self.obj == 'aec':
                    def aec_val(raw_scores, y_true):
                        scores = expit(raw_scores)

                        # Return AEC (not grad/hess)
                        # ec = (1 - scores) * y_val * amounts_val + scores * fixed_cost
                        ec = y_true * (
                            scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1]) \
                            + (1 - y_true) * (
                            scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                        aec = ec.mean()

                        return 'AEC', aec  # Todo: do not return 'AEC'

                    aec = aec_val(scores, y_val)
                    val_loss = aec[1]
                print('\t\tLambda l1 = %.4f;\tLoss = %.5f' % (lambda1, val_loss))
                losses_list.append(val_loss)
            lambda1_opt = lambda1_list[np.argmin(losses_list)]
            print('\tOptimal lambda = %.4f' % lambda1_opt)
            self.params['reg_alpha'] = lambda1_opt
        elif l2:
            self.params['reg_alpha'] = 0
            losses_list = []
            for lambda2 in lambda2_list:
                xgboost = CSBoost(obj=self.obj, lambda2=lambda2)
                xgboost = xgboost.fit(x_train, y_train, x_val, y_val, cost_matrix_train, cost_matrix_val)
                scores = xgboost.inplace_predict(x_val)

                # Evaluate loss (without regularization term!)
                if self.obj == 'ce':
                    eps = 1e-9  # small value to avoid log(0)
                    ce = - (y_val * np.log(scores + eps) + (1 - y_val) * np.log(1 - scores + eps))
                    val_loss = ce.mean()
                elif self.obj == 'weightedce':
                    eps = 1e-9  # small value to avoid log(0)
                    ce = - (y_val * np.log(scores + eps) + (1 - y_val) * np.log(1 - scores + eps))

                    cost_misclass = np.zeros(len(y_val))
                    cost_misclass[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                    cost_misclass[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                    weighted_ce = cost_misclass * ce
                    val_loss = weighted_ce.mean()
                elif self.obj == 'aec':
                    def aec_val(raw_scores, y_true):
                        scores = expit(raw_scores)

                        # Return AEC (not grad/hess)
                        ec = y_true * (
                                scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1]) \
                             + (1 - y_true) * (
                                     scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                        aec = ec.mean()

                        return 'AEC', aec

                    aec = aec_val(scores, y_val)
                    val_loss = aec[1]
                print('\t\tLambda l2 = %.4f;\tLoss = %.5f' % (lambda2, val_loss))
                losses_list.append(val_loss)
            lambda2_opt = lambda2_list[np.argmin(losses_list)]
            print('\tOptimal lambda = %.4f' % lambda2_opt)
            self.params['reg_alpha'] = lambda2_opt
        else:
            self.lambda1 = 0
            self.lambda2 = 0
