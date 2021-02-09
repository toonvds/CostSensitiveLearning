import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import spearmanr, combine_pvalues, friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from tabulate import tabulate

from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
hmeasure = importr('hmeasure')
rpy2.robjects.numpy2ri.activate()

# Matplotlib settings for figures:
#plt.style.use('science')
plt.rcParams.update({'font.size': 14})
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rcParams['figure.figsize'] = (7, 6)
plt.rcParams['figure.dpi'] = 250


def get_performance_metrics(evaluators, evaluation_matrices, i, index, cost_matrix, labels, probabilities, predictions):
    if evaluators['traditional']:
        true_pos = (predictions * labels).sum()
        true_neg = ((1-predictions) * (1-labels)).sum()
        false_pos = (predictions * (1-labels)).sum()
        false_neg = ((1-predictions) * labels).sum()

        # TODO: Make sure no division by 0! Especially precision
        accuracy = (true_pos + true_neg) / len(labels)
        recall = true_pos / (true_pos + false_neg)
        precision = true_pos / (true_pos + false_pos)
        f1_score = 2 * (precision * recall) / (precision + recall)

        evaluation_matrices['traditional'][index, i] = np.array([accuracy, recall, precision, f1_score])

    if evaluators['ROC']:
        fpr, tpr, roc_thresholds = metrics.roc_curve(y_true=labels, y_score=probabilities)

        evaluation_matrices['ROC'][index, i] = np.array([fpr, tpr, roc_thresholds])

    if evaluators['AUC']:
        auc = metrics.roc_auc_score(y_true=labels, y_score=probabilities)

        evaluation_matrices['AUC'][index, i] = auc

    if evaluators['savings']:
        cost_without = cost_without_algorithm(cost_matrix, labels)
        cost_with = cost_with_algorithm(cost_matrix, labels, predictions)
        savings = 1 - cost_with / cost_without

        evaluation_matrices['savings'][index, i] = savings

    if evaluators['ROCIV']:
        # TODO: sample weights with weights from cost_matrix? See PRIV

        misclass_costs = np.zeros(len(labels))
        misclass_costs[labels == 0] = cost_matrix[:, 1, 0][labels == 0]
        misclass_costs[labels == 1] = cost_matrix[:, 0, 1][labels == 1]
        fpcosts, tpbenefits, auciv = rociv(labels, probabilities, misclass_costs)

        evaluation_matrices['ROCIV'][index, i] = np.array([fpcosts, tpbenefits, auciv], dtype=object)

    if evaluators['H_measure']:
        # TODO:
        #   Takes approx 1 sec -> Do from scratch?
        #   https://github.com/canagnos/hmeasure-python/blob/master/mcp/mcp.py
        #   Specify cost distribution?
        #   Uses hmeasure, see https://github.com/cran/hmeasure/blob/master/R/library_metrics.R

        misclass_neg = cost_matrix[:, 1, 0][labels == 0]
        misclass_pos = cost_matrix[:, 0, 1][labels == 1]

        # Todo: explain this severity!
        severity = misclass_neg.mean() / misclass_pos.mean()

        h = hmeasure.HMeasure(labels, probabilities[:, None], severity)[0][0][0]

        evaluation_matrices['H_measure'][index, i] = h

    if evaluators['PR']:
        precision, recall, _ = metrics.precision_recall_curve(y_true=labels, probas_pred=probabilities)

        # AUC is not recommended here (see sklearn docs)
        # We will use Average Precision (AP)
        ap = metrics.average_precision_score(y_true=labels, y_score=probabilities)

        evaluation_matrices['PR'][index, i] = np.array([precision, recall, ap], dtype=object)

    if evaluators['PRIV']:
        misclass_costs = np.zeros(len(labels))
        misclass_costs[labels == 0] = cost_matrix[:, 1, 0][labels == 0]
        misclass_costs[labels == 1] = cost_matrix[:, 0, 1][labels == 1]

        precisioniv, recalliv, _ = metrics.precision_recall_curve(y_true=labels, probas_pred=probabilities,
                                                                  sample_weight=misclass_costs)

        # AUC is not recommended here (see sklearn docs)
        # We will use Average Precision (AP)
        apiv = metrics.average_precision_score(y_true=labels, y_score=probabilities, sample_weight=misclass_costs)
        #ap = metrics.auc(recall, precision)

        evaluation_matrices['PRIV'][index, i] = np.array([precisioniv, recalliv, apiv], dtype=object)

    if evaluators['rankings']:

        pos_probas = probabilities[labels == 1]
        # Get C_(0,1) (FN - misclassification cost of positive instances)
        misclass_costs_pos = cost_matrix[:, 0, 1][labels == 1]

        # Sort indices from high to low
        sorted_indices_probas = np.argsort(pos_probas)[::-1]
        prob_rankings = np.argsort(sorted_indices_probas)

        sorted_indices_amounts = np.argsort(misclass_costs_pos)[::-1]
        amount_rankings = np.argsort(sorted_indices_amounts)

        #  Compare rankings of probas with rankings of amounts for all positive instances
        spearman_test = spearmanr(prob_rankings, amount_rankings)
        #spearman_test = spearmanr(pos_probas[sorted_indices_probas], pos_amounts[sorted_indices_probas])
        #spearman_test = spearmanr(probabilities, amounts)

        evaluation_matrices['rankings'][index, i] = np.array([misclass_costs_pos[sorted_indices_probas], spearman_test],
                                                             dtype=object)

    if evaluators['brier']:

        brier = ((probabilities - labels)**2).mean()

        evaluation_matrices['brier'][index, i] = brier

    return evaluation_matrices


def cost_with_algorithm(cost_matrix, labels, predictions):

    cost_tn = cost_matrix[:, 0, 0][np.logical_and(predictions == 0, labels == 0)].sum()
    cost_fn = cost_matrix[:, 0, 1][np.logical_and(predictions == 0, labels == 1)].sum()
    cost_fp = cost_matrix[:, 1, 0][np.logical_and(predictions == 1, labels == 0)].sum()
    cost_tp = cost_matrix[:, 1, 1][np.logical_and(predictions == 1, labels == 1)].sum()

    return sum((cost_tn, cost_fn, cost_fp, cost_tp))


def cost_without_algorithm(cost_matrix, labels):

    # Predict everything as the default class that leads to minimal cost
    # Also include cost of TP/TN!
    cost_neg = cost_matrix[:, 0, 0][labels == 0].sum() + cost_matrix[:, 0, 1][labels == 1].sum()
    cost_pos = cost_matrix[:, 1, 0][labels == 0].sum() + cost_matrix[:, 1, 1][labels == 1].sum()

    return min(cost_neg, cost_pos)


def rociv(labels, probabilities, costs):
    # Total cost per class
    pos_total = costs[labels == 1].sum()
    neg_total = costs[labels == 0].sum()

    # Sort predictions (1 to 0) and corresponding labels
    sorted_indices = np.argsort(probabilities)[::-1]
    costs_sorted = costs[sorted_indices]
    labels_sorted = labels[sorted_indices]
    # probabilities[sorted_indices][labels_sorted == 1]

    # Create ROCIV curve
    fp_costs = [0]
    tp_benefits = [0]

    benefits_cum = 0
    costs_cum = 0

    for i in range(len(labels)):
        if labels_sorted[i]:
            benefits_cum += costs_sorted[i]
        else:
            costs_cum += costs_sorted[i]

        fp_costs.append(costs_cum / neg_total)
        tp_benefits.append(benefits_cum / pos_total)

    # Area under curve
    auciv = metrics.auc(x=fp_costs, y=tp_benefits)
    # auciv = np.trapz(y=tp_benefits, x=fp_costs)

    return fp_costs, tp_benefits, auciv


def evaluate_experiments(evaluators, methodologies, evaluation_matrices, directory, name):
    table_evaluation = []
    n_methodologies = sum(methodologies.values())

    if evaluators['traditional']:

        table_traditional = [['Method', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'AR']]

        # Compute F1 rankings (- as higher is better)
        all_f1s = []
        for i in range(evaluation_matrices['traditional'].shape[0]):
            method_f1s = []
            for j in range(evaluation_matrices['traditional'][i].shape[0]):
                f1 = evaluation_matrices['traditional'][i][j][-1]
                method_f1s.append(f1)
            all_f1s.append(np.array(method_f1s))
        ranked_args = np.argsort(-np.array(all_f1s), axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = np.mean(rankings, axis=1)

        # Summarize all per method
        index = 0
        for item, value in methodologies.items():
            if value:
                averages = evaluation_matrices['traditional'][index, :].mean()

                table_traditional.append([item, averages[0], averages[1], averages[2], averages[3],
                                          avg_rankings[index]])

                index += 1

        print(tabulate(table_traditional, headers="firstrow", floatfmt=("", ".4f", ".4f", ".4f", ".4f", ".2f")))
        table_evaluation.append(table_traditional)

        # Do tests if enough measurements are available (at least 3)
        if np.array(all_f1s).shape[1] > 2:
            friedchisq = friedmanchisquare(*np.transpose(all_f1s))
            print('\nF1 - Friedman test')
            print('H0: Model performance follows the same distribution')
            print('\tChi-square:\t%.4f' % friedchisq[0])
            print('\tp-value:\t%.4f' % friedchisq[1])
            if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                # Post-hoc Nemenyi Friedman: Rows are blocks, columns are groups
                nemenyi = posthoc_nemenyi_friedman(np.array(all_f1s).T.astype(dtype=np.float32))
                print('\nNemenyi post hoc test:')
                print(nemenyi)
            print('_________________________________________________________________________')

    if evaluators['ROC']:
        index = 0
        fig, ax = plt.subplots()
        ax.set_title('ROC curve')
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        for item, value in methodologies.items():
            if value:
                # See https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
                tprs = []
                mean_fpr = np.linspace(0, 1, 100)

                for i in range(evaluation_matrices['ROC'][index, :].shape[0]):
                    fpr, tpr, _ = list(evaluation_matrices['ROC'][index, i])
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)

                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0

                index += 1

                ax.plot(mean_fpr, mean_tpr, label=item, lw=2, alpha=.8)

                # std_tpr = np.std(tprs, axis=0)
                # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)

        ax.legend()
        plt.savefig(str(directory + 'ROC.png'), bbox_inches='tight')
        plt.show()

    if evaluators['AUC']:

        table_auc = [['Method', 'AUC', 'sd', 'AR']]

        # Compute rankings (higher is better)
        ranked_args = (-evaluation_matrices['AUC']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_auc.append([item, evaluation_matrices['AUC'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['AUC'][index, :].var()), avg_rankings[index]])
                index += 1

        print(tabulate(table_auc, headers="firstrow", floatfmt=("", ".4f", ".4f", ".2f")))
        table_evaluation.append(table_auc)

        # Do tests if enough measurements are available (at least 3)
        if evaluation_matrices['AUC'].shape[1] > 2:
            friedchisq = friedmanchisquare(*evaluation_matrices['AUC'].T)
            print('\nAUC - Friedman test')
            print('H0: Model performance follows the same distribution')
            print('\tChi-square:\t%.4f' % friedchisq[0])
            print('\tp-value:\t%.4f' % friedchisq[1])
            if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                nemenyi = posthoc_nemenyi_friedman(evaluation_matrices['AUC'].T.astype(dtype=np.float32))
                print('\nNemenyi post hoc test:')
                print(nemenyi)
            print('_________________________________________________________________________')

    if evaluators['savings']:

        table_savings = [['Method', 'Savings', 'sd', 'AR']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['savings']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)

        # Summarize per method
        index = 0
        methods_used = []
        for item, value in methodologies.items():
            if value:
                methods_used.append(item)
                table_savings.append([item, evaluation_matrices['savings'][index, :].mean(),
                                      np.sqrt(evaluation_matrices['savings'][index, :].var()), avg_rankings[index]])
                index += 1

        print(tabulate(table_savings, headers="firstrow", floatfmt=("", ".4f", ".4f", ".2f")))
        table_evaluation.append(table_savings)

        plt.xlabel('Methods')
        plt.ylabel('Savings')
        # plt.ylim(0, 1)
        plt.boxplot(np.transpose(evaluation_matrices['savings']))
        plt.xticks(np.arange(n_methodologies) + 1, methods_used)
        plt.xticks(rotation=40)
        plt.savefig(str(directory + 'savings_boxplot_' + name + '.png'), bbox_inches='tight')
        plt.show()

        # Do tests if enough measurements are available (at least 3)
        if evaluation_matrices['savings'].shape[1] > 2:
            friedchisq = friedmanchisquare(*evaluation_matrices['savings'].T)
            print('\nSavings - Friedman test')
            print('H0: Model performance follows the same distribution')
            print('\tChi-square:\t%.4f' % friedchisq[0])
            print('\tp-value:\t%.4f' % friedchisq[1])
            if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                nemenyi = posthoc_nemenyi_friedman(evaluation_matrices['savings'].T.astype(dtype=np.float32))
                print('\nNemenyi post hoc test:')
                print(nemenyi)
            print('_________________________________________________________________________')

    if evaluators['ROCIV']:
        table_auciv = [['Method', 'AUCIV', 'sd', 'AR']]

        index = 0
        fig2, ax2 = plt.subplots()
        ax2.set_title('ROCIV curve')
        ax2.set_xlabel('False positive cost')
        ax2.set_ylabel('True positive benefit')

        all_aucivs = []

        for item, value in methodologies.items():
            if value:
                # See https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
                tprs = []
                mean_fpr = np.linspace(0, 1, 100)
                aucivs = []

                for i in range(evaluation_matrices['ROCIV'][index, :].shape[0]):
                    fp_costs, tp_benefits, auciv = list(evaluation_matrices['ROCIV'][index, i])
                    interp_tpr = np.interp(mean_fpr, fp_costs, tp_benefits)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    aucivs.append(auciv)

                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0

                ax2.plot(mean_fpr, mean_tpr, label=item, lw=2, alpha=.8)
                # std_tpr = np.std(tprs, axis=0)
                # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                # ax2.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)

                aucivs = np.array(aucivs)
                table_auciv.append([item, aucivs.mean(), np.sqrt(aucivs.var())])
                all_aucivs.append(aucivs)

                index += 1

        ax2.legend()
        plt.savefig(str(directory + 'ROCIV.png'), bbox_inches='tight')
        plt.show()

        # Add rankings (higher is better)
        ranked_args = np.argsort(-np.array(all_aucivs), axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = np.mean(rankings, axis=1)
        for i in range(1, len(table_auciv)):
            table_auciv[i].append(avg_rankings[i-1])

        print(tabulate(table_auciv, headers="firstrow", floatfmt=("", ".4f", ".4f", ".2f")))
        table_evaluation.append(table_auciv)

        # Do tests if enough measurements are available (at least 3)
        if np.array(all_aucivs).shape[1] > 2:
            friedchisq = friedmanchisquare(*np.transpose(all_aucivs))
            print('\nAUCIV - Friedman test')
            print('H0: Model performance follows the same distribution')
            print('\tChi-square:\t%.4f' % friedchisq[0])
            print('\tp-value:\t%.4f' % friedchisq[1])
            if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                nemenyi = posthoc_nemenyi_friedman(np.array(all_aucivs).T.astype(dtype=np.float32))
                print('\nNemenyi post hoc test:')
                print(nemenyi)
            print('_________________________________________________________________________')

    if evaluators['H_measure']:

        table_H = [['Method', 'H_measure', 'sd', 'AR']]

        # Compute rankings (- as higher is better)
        ranked_args = (-evaluation_matrices['H_measure']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_H.append([item, evaluation_matrices['H_measure'][index, :].mean(),
                                np.sqrt(evaluation_matrices['H_measure'][index, :].var()), avg_rankings[index]])
                index += 1

        print(tabulate(table_H, headers="firstrow", floatfmt=("", ".4f", ".4f", ".2f")))
        table_evaluation.append(table_H)

        # Do tests if enough measurements are available (at least 3)
        if evaluation_matrices['H_measure'].shape[1] > 2:
            friedchisq = friedmanchisquare(*evaluation_matrices['H_measure'].T)
            print('\nH-measure - Friedman test')
            print('H0: Model performance follows the same distribution')
            print('\tChi-square:\t%.4f' % friedchisq[0])
            print('\tp-value:\t%.4f' % friedchisq[1])
            if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                nemenyi = posthoc_nemenyi_friedman(evaluation_matrices['H_measure'].T.astype(dtype=np.float32))
                print('\nNemenyi post hoc test:')
                print(nemenyi)
            print('_________________________________________________________________________')

    if evaluators['PR']:
        table_ap = [['Method', 'Avg Prec', 'sd', 'AR']]

        index = 0
        fig2, ax2 = plt.subplots()
        ax2.set_title('PR curve')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')

        all_aps = []
        for item, value in methodologies.items():
            if value:
                # See https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
                precisions = []
                mean_recall = np.linspace(0, 1, 100)
                aps = []

                for i in range(evaluation_matrices['PR'][index, :].shape[0]):
                    precision, recall, ap = list(evaluation_matrices['PR'][index, i])

                    interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
                    interp_precision[0] = 1
                    precisions.append(interp_precision)

                    aps.append(ap)

                mean_precision = np.mean(precisions, axis=0)
                mean_precision[-1] = 0

                ax2.plot(mean_recall, mean_precision, label=item, lw=2, alpha=.8)
                # std_precision = np.std(precisions, axis=0)
                # precisions_upper = np.minimum(mean_precision + std_precision, 1)
                # precisions_lower = np.maximum(mean_precision - std_precision, 0)
                # ax2.fill_between(mean_recall, precisions_lower, precisions_upper, color='grey', alpha=.2)

                aps = np.array(aps)
                table_ap.append([item, aps.mean(), np.sqrt(aps.var())])

                all_aps.append(aps)

                index += 1

        ax2.legend()
        plt.savefig(str(directory + 'PR.png'), bbox_inches='tight')
        plt.show()

        # Add rankings
        ranked_args = np.argsort(-np.array(all_aps), axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = np.mean(rankings, axis=1)
        for i in range(1, len(table_ap)):
            table_ap[i].append(avg_rankings[i - 1])

        print(tabulate(table_ap, headers="firstrow", floatfmt=("", ".4f", ".4f", ".2f")))
        table_evaluation.append(table_ap)

        # Do tests if enough measurements are available (at least 3)
        if np.array(all_aps).shape[1] > 2:
            friedchisq = friedmanchisquare(*np.transpose(all_aps))
            print('\nAP - Friedman test')
            print('H0: Model performance follows the same distribution')
            print('\tChi-square:\t%.4f' % friedchisq[0])
            print('\tp-value:\t%.4f' % friedchisq[1])
            if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                nemenyi = posthoc_nemenyi_friedman(np.array(all_aps).T.astype(dtype=np.float32))
                print('\nNemenyi post hoc test:')
                print(nemenyi)
            print('_________________________________________________________________________')

    if evaluators['PRIV']:
        table_apiv = [['Method', 'APIV', 'sd', 'AR']]

        index = 0
        fig2, ax2 = plt.subplots()
        ax2.set_title('PRIV curve')
        ax2.set_xlabel('Weighted recall')
        ax2.set_ylabel('Weighted precision')

        all_apivs = []
        for item, value in methodologies.items():
            if value:
                # See https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
                precisions = []
                mean_recall = np.linspace(0, 1, 100)
                apivs = []

                for i in range(evaluation_matrices['PRIV'][index, :].shape[0]):
                    precision, recall, apiv = list(evaluation_matrices['PRIV'][index, i])

                    interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
                    interp_precision[0] = 1
                    precisions.append(interp_precision)

                    apivs.append(apiv)

                mean_precision = np.mean(precisions, axis=0)
                mean_precision[-1] = 0

                ax2.plot(mean_recall, mean_precision, label=item, lw=2, alpha=.8)
                # std_precision = np.std(precisions, axis=0)
                # precisions_upper = np.minimum(mean_precision + std_precision, 1)
                # precisions_lower = np.maximum(mean_precision - std_precision, 0)
                # ax2.fill_between(mean_recall, precisions_lower, precisions_upper, color='grey', alpha=.2)

                apivs = np.array(apivs)
                table_apiv.append([item, apivs.mean(), np.sqrt(apivs.var())])
                all_apivs.append(apivs)

                index += 1

        ax2.legend()
        plt.savefig(str(directory + 'PRIV.png'), bbox_inches='tight')
        plt.show()

        # Add rankings
        ranked_args = np.argsort(-np.array(all_apivs), axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = np.mean(rankings, axis=1)
        for i in range(1, len(table_apiv)):
            table_apiv[i].append(avg_rankings[i - 1])

        print(tabulate(table_apiv, headers="firstrow", floatfmt=("", ".4f", ".4f", ".2f")))
        table_evaluation.append(table_apiv)

        # Do tests if enough measurements are available (at least 3)
        if np.array(all_apivs).shape[1] > 2:
            friedchisq = friedmanchisquare(*np.transpose(all_apivs))
            print('\nAPIV - Friedman test')
            print('H0: Model performance follows the same distribution')
            print('\tChi-square:\t%.4f' % friedchisq[0])
            print('\tp-value:\t%.4f' % friedchisq[1])
            if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                nemenyi = posthoc_nemenyi_friedman(np.array(all_apivs).T.astype(dtype=np.float32))
                print('\nNemenyi post hoc test:')
                print(nemenyi)
            print('_________________________________________________________________________')

    if evaluators['rankings']:
        table_spearman = [['Method', 'Spearman corr', 'p-value', 'AR']]

        index = 0
        fig3, ax3 = plt.subplots()
        ax3.set_title('Rankings')
        ax3.set_xlabel('% positives included')
        ax3.set_ylabel('% of total amounts')

        all_scs = []
        for item, value in methodologies.items():
            if value:
                # See https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
                total_cum_frac_amounts = []
                mean_included = np.linspace(0, 1, 100)
                spearman_corrs = []
                p_values = []

                for i in range(evaluation_matrices['rankings'][index, :].shape[0]):
                    ranked_amounts, spearman_test = list(evaluation_matrices['rankings'][index, i])
                    spearman_corr, p_value = spearman_test

                    cum_frac_amounts = np.cumsum(ranked_amounts) / ranked_amounts.sum()

                    interp_cum_frac_amounts = np.interp(mean_included,
                                                        np.arange(0, len(cum_frac_amounts)) / len(cum_frac_amounts),
                                                        cum_frac_amounts)
                    total_cum_frac_amounts.append(interp_cum_frac_amounts)

                    spearman_corrs.append(spearman_corr)
                    p_values.append(p_value)

                mean_cum_amounts = np.mean(total_cum_frac_amounts, axis=0)

                ax3.plot(mean_cum_amounts, label=item, lw=2, alpha=.8)
                # std_precision = np.std(precisions, axis=0)
                # precisions_upper = np.minimum(mean_precision + std_precision, 1)
                # precisions_lower = np.maximum(mean_precision - std_precision, 0)
                # ax2.fill_between(mean_recall, precisions_lower, precisions_upper, color='grey', alpha=.2)

                spearman_corrs = np.array(spearman_corrs)
                # p_values = np.array(p_values)
                # Todo: Fisher's method right one?
                p_value_total = combine_pvalues(p_values)[1]  # Take only the p-value (not statistic)
                table_spearman.append([item, spearman_corrs.mean(), p_value_total])

                all_scs.append(spearman_corrs)

                index += 1

        best_amounts = np.cumsum(np.sort(ranked_amounts)[::-1] / ranked_amounts.sum())
        best_amounts_100 = np.interp(mean_included, np.arange(0, len(best_amounts)) / len(best_amounts), best_amounts)
        ax3.plot(best_amounts_100, linestyle='dashed', color='k', label='best possible')
        ax3.plot(np.cumsum(np.ones(100) / 100), linestyle='dotted', color='k', label='random')
        # Todo: Does this correspond to a "random" policy? Yes - average...

        # Add rankings
        ranked_args = np.argsort(-np.array(all_scs), axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = np.mean(rankings, axis=1)
        for i in range(1, len(table_spearman)):
            table_spearman[i].append(avg_rankings[i - 1])

        ax3.legend(bbox_to_anchor=(1.04,1), prop={'size': 12})
        fig3.set_size_inches((7, 4))

        plt.savefig(str(directory + 'rankings.png'), bbox_inches='tight')
        plt.show()

        print(tabulate(table_spearman, headers="firstrow", floatfmt=("", ".4f", ".4f", ".2f")))
        table_evaluation.append(table_spearman)

        # Do tests if enough measurements are available (at least 3)
        if np.array(all_scs).shape[1] > 2:
            friedchisq = friedmanchisquare(*np.transpose(all_scs))
            print('\nSpearman correlation - Friedman test')
            print('H0: Model performance follows the same distribution')
            print('\tChi-square:\t%.4f' % friedchisq[0])
            print('\tp-value:\t%.4f' % friedchisq[1])
            if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                nemenyi = posthoc_nemenyi_friedman(np.array(all_scs).T.astype(dtype=np.float32))
                print('\nNemenyi post hoc test:')
                print(nemenyi)
            print('_________________________________________________________________________')

    if evaluators['brier']:
        table_brier = [['Method', 'Brier score', 'sd', 'AR']]

        # Compute rankings (higher is better)  # Todo: higher is not better!!!
        ranked_args = (-evaluation_matrices['brier']).argsort(axis=0)
        rankings = np.arange(len(ranked_args))[ranked_args.argsort(axis=0)]
        rankings = rankings + 1
        avg_rankings = rankings.mean(axis=1)

        # Summarize per method
        index = 0
        for item, value in methodologies.items():
            if value:
                table_brier.append([item, evaluation_matrices['brier'][index, :].mean(),
                                  np.sqrt(evaluation_matrices['brier'][index, :].var()), avg_rankings[index]])
                index += 1

        print(tabulate(table_brier, headers="firstrow", floatfmt=("", ".6f", ".6f", ".2f")))
        table_evaluation.append(table_brier)

        # Do tests if enough measurements are available (at least 3)
        if evaluation_matrices['brier'].shape[1] > 2:
            friedchisq = friedmanchisquare(*evaluation_matrices['brier'].T)
            print('\nBrier score - Friedman test')
            print('H0: Model performance follows the same distribution')
            print('\tChi-square:\t%.4f' % friedchisq[0])
            print('\tp-value:\t%.4f' % friedchisq[1])
            if friedchisq[1] < 0.05:  # If p-value is significant, do Nemenyi post hoc test
                nemenyi = posthoc_nemenyi_friedman(evaluation_matrices['brier'].T.astype(dtype=np.float32))
                print('\nNemenyi post hoc test:')
                print(nemenyi)
            print('_________________________________________________________________________')

    if evaluators['LaTeX']:
        top_row = ['Method']
        n_methods = len(table_evaluation[0]) - 1
        bottom_rows = [None] * n_methods
        # Print to LaTeX:
        for i in range(len(table_evaluation)):
            # Print tables separately
            print(tabulate(table_evaluation[i], headers="firstrow", floatfmt=".4f", tablefmt='latex'))
            for j in range(len(table_evaluation[i])):
                if j == 0:
                    top_row += table_evaluation[i][j][1:]
                else:  # j>0
                    if i == 0:
                        bottom_rows[j - 1] = [table_evaluation[i][j][0]]
                    bottom_rows[j - 1] += table_evaluation[i][j][1:]
        total_table = [top_row] + bottom_rows

        print('\n' + tabulate(total_table, headers="firstrow", floatfmt=".4f"))
        print('\n' + tabulate(total_table, headers="firstrow", floatfmt=".4f", tablefmt='latex'))

    # Add to summary file
    with open(str(directory + 'summary.txt'), 'a') as file:
        for i in table_evaluation:
            file.write(tabulate(i, floatfmt=".4f"))

        if evaluators['LaTeX']:
            file.write(tabulate(total_table, headers="firstrow", floatfmt=".4f", tablefmt='latex'))
