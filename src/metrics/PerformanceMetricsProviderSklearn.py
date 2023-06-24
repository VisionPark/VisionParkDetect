import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn import metrics
from src.metrics.PerformanceMetricsProvider import PerformanceMetricsProvider
import pandas as pd

from src.metrics.entity.PerformanceMetrics import PerformanceMetrics

from sklearn.metrics import roc_auc_score, auc, average_precision_score
from adjustText import adjust_text
from matplotlib.font_manager import FontProperties
import numpy as np
from numpy import trapz
import concurrent.futures
from scipy.spatial import distance
# https://www.iartificial.net/precision-recall-f1-accuracy-en-clasificacion/
# Precision nos da la calidad de la predicción: ¿qué porcentaje de los que hemos dicho que son la clase positiva, en realidad lo son?
# Sensibilidad o Recall nos da la cantidad: ¿qué porcentaje de la clase positiva hemos sido capaces de identificar?
# F1 combina Precision y Recall en una sola medida
# La Matriz de Confusión indica qué tipos de errores se cometen


class PerformanceMetricsProviderSklearn(PerformanceMetricsProvider):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_metrics(self):
        tn, fp, fn, tp = metrics.confusion_matrix(
            self.real, self.predicted, labels=[False, True]).ravel()

        # Precision (or Sensivity) Score = TP / (FP + TP). Minimize FP
        precision = tp / (fp+tp) if fp+tp != 0 else 1.0

        # Specificity score = TN / (TN+FP)
        # False positive rate = 1 - Specificity
        specificity = tn / (tn+fp)

        # True positive rate
        # Sensibilidad o Recall Score = TP / (FN + TP). Minimize FN
        recall = tp / (fn+tp)

        # F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/) . Minimize FN over minimizing FP
        f1 = 2*precision*recall / (precision + recall)

        # Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
        accuracy = (tp+tn) / (tp+fn+tn+fp)

        # Update metrics object
        self.metrics.precision = precision
        self.metrics.specificity = specificity
        self.metrics.recall = recall
        self.metrics.f1 = f1
        self.metrics.accuracy = accuracy

    # Override abstract method

    def show_confusion_matrix(self, plot=True):
        print('Precision: %.3f' % self.metrics.precision)
        print('specificity: %.3f' % self.metrics.specificity)
        print('Recall: %.3f' % self.metrics.recall)
        print('F1 Score: %.3f' % self.metrics.f1)
        print('Accuracy: %.3f' % self.metrics.accuracy)

        if (plot):
            cm_display = metrics.ConfusionMatrixDisplay(
                confusion_matrix=metrics.confusion_matrix(
                    self.real, self.predicted),
                display_labels=['Occupied', 'Vacant'])
            cm_display.plot()
            plt.show()

    @staticmethod
    def show_dataframe(metrics_list: list[PerformanceMetrics], row_names: list[str]):
        precision_total = [
            metrics_list[i].precision for i in range(len(metrics_list))]
        specificity_total = [
            metrics_list[i].specificity for i in range(len(metrics_list))]
        recall_total = [
            metrics_list[i].recall for i in range(len(metrics_list))]
        f1_total = [metrics_list[i].f1 for i in range(len(metrics_list))]
        accuracy_total = [
            metrics_list[i].accuracy for i in range(len(metrics_list))]

        data = {
            "precision": precision_total,
            "specificity": specificity_total,
            "recall": recall_total,
            "f1": f1_total,
            "accuracy": accuracy_total
        }

        df = pd.DataFrame(data, index=row_names)
        print(df.round(3))

    @staticmethod
    def show_tpr_fpr(metrics_dict_vt: dict, metrics_extra: dict = None, show_diff=False, save_fig_dst=None, first_param_str="blockSize"):

        fontP = FontProperties(family='sans-serif', size=8)
        # fontP.set_size('xx-small')

        fig, ax = plt.subplots(2, 1, figsize=(5, 10), dpi=300, sharey=True)

        index = 0

        def process_metrics_dict_diff(vt, metrics_dict_diff):
            tpr_list = list()
            fpr_list = list()
            precision_list = list()

            for diff, metrics in metrics_dict_diff.items():
                tpr_list.append(metrics.recall)
                fpr_list.append(1 - metrics.specificity)
                precision_list.append(metrics.precision)

            auc = trapz(x=fpr_list, y=tpr_list)
            ap = trapz(x=tpr_list, y=precision_list)

            # ROC CURVE
            li = zip(*[fpr_list, tpr_list])
            ax[0].plot(*zip(*li), linestyle='--', marker='.',
                       label=f'{index}: {first_param_str}={vt} AUC={auc:.3f}')

            # PR CURVE
            li_recall = zip(*[tpr_list, precision_list])
            ax[1].plot(*zip(*li_recall), linestyle='--', marker='.',
                       label=f'{index}: {first_param_str}={vt} AP={ap:.3f}')

            # return diff_list

        for vt, metrics_dict_diff in metrics_dict_vt.items():
            process_metrics_dict_diff(vt, metrics_dict_diff)
            index += 1

        # Optional: plot extra metrics
        if metrics_extra is not None:
            for label, metrics in metrics_extra.items():
                tpr = metrics.recall
                fpr = 1 - metrics.specificity
                precision = metrics.precision

                # ROC CURVE
                # li = zip(*[fpr, tpr])
                ax[0].plot((fpr, tpr), marker='.',
                           label=f'{index}: {label}')

                # PR CURVE
                # li_recall = zip(*[tpr, precision])
                ax[1].plot((tpr, precision), linestyle='--', marker='.',
                           label=f'{index}: {label}')

                index += 1

        # Set fixed x and y axis limits and ticks
        for axi in ax:
            axi.set_xlim([-0.025, 1.025])
            axi.set_ylim([-0.025, 1.025])
            axi.set_xticks(np.arange(0, 1.1, 0.1))
            axi.set_yticks(np.arange(0, 1.05, 0.05))
            axi.xaxis.set_major_locator(plt.MultipleLocator(0.1))
            axi.yaxis.set_major_locator(plt.MultipleLocator(0.05))
            axi.grid(color='gray', linestyle='--', linewidth=0.5)
            axi.legend(loc='lower right', prop=fontP)
            axi.tick_params(axis='both', labelsize=6)

        # Graph 1: FPR-TPR
        plt.subplot(2, 1, 1)
        # fig.suptitle(f'{parking_id}-{weather}')
        ax[0].set_title('ROC Curve')
        ax[0].set_xlabel('False positive rate (1 - Specificity)', fontsize=8)
        ax[0].set_ylabel('True positive rate (Recall)', fontsize=8)

        plt.plot(0, 1, marker='x',
                 label='Perfect classifier')

        # if show_diff:
        #     texts = [plt.text(fpr_list[i], tpr_list[i], diff_list[i], size=8)
        #              for i in range(len(fpr_list))]
        #     adjust_text(texts, arrowprops={
        #                 'arrowstyle': '->'}, expand_points=(1.1, 2), x=all_fpr_list, y=all_tpr_list)

        # Graph 2: Recall-TPR
        plt.subplot(2, 1, 2)
        ax[1].set_title('PR Curve')
        ax[1].set_xlabel('True positive rate (Recall)', fontsize=8)
        ax[1].set_ylabel('Precision', fontsize=8)

        plt.plot(1, 1, marker='x',
                 label='Perfect classifier')

        # if show_diff:
        #     texts = [plt.text(tpr_list[i], precision_list[i], diff_list[i], size=8)
        #              for i in range(len(fpr_list))]
        #     adjust_text(texts, arrowprops={
        #                 'arrowstyle': '->'}, expand_points=(1.1, 2), x=all_tpr_list, y=all_precision_list)

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=1,
                            top=0.9,
                            wspace=0.1,
                            hspace=0.2)

        if save_fig_dst is not None:
            plt.savefig(save_fig_dst, facecolor='white', bbox_inches='tight')

        plt.show()

    @staticmethod
    def choose_parameters(metrics_dict: dict, first_param_str="bs"):
        """
        Calculates the AUC and AP for all curves in the given nested dictionary, along with the average for each weather condition and for all conditions combined.

        Args:
        - metrics_dict: A nested dictionary containing performance metrics for different combinations of weather condition, blockSize and vacantTreshold.
                        The most outer dictionary has string keys for weather condition and the next inner dicitonary as keys.
                        The next outer dictionary has keys representing block_size values and inner dictionary as keys
                        representing vacant_threshold values. The values in the inner dictionary are instances of PerformanceMetrics.
        Returns:
        - A pandas dataframe containing the AUC, AP, and average metrics for each weather condition and for all conditions combined.
        """

        # bs_auc_dict = dict()
        # bs_ap_dict = dict()

        roc_ideal = [0, 1]  # ideal point for ROC curve
        pr_ideal = [1, 1]  # ideal point for PR curve
        best_metrics = []  # list to store the bestmetrics
        distances_roc = []  # list to store the distances to the ideal points
        distances_pr = []  # list to store the distances to the ideal points

        # max_auc = -1
        max_auc_weather = dict()  # (AUC, bs)

        for weather_condition, weather_dict in metrics_dict.items():

            max_auc_weather[weather_condition] = [-1, -1]

            for bs, vacant_dict in weather_dict.items():
                tpr_list = list()
                fpr_list = list()
                precision_list = list()
                for vt, metrics in vacant_dict.items():

                    tpr_list.append(metrics.recall)
                    fpr_list.append(1 - metrics.specificity)
                    precision_list.append(metrics.precision)

                auc = trapz(x=fpr_list, y=tpr_list)
                ap = trapz(x=tpr_list, y=precision_list)

                if auc > max_auc_weather[weather_condition][0]:
                    max_auc_weather[weather_condition][0] = auc
                    max_auc_weather[weather_condition][1] = bs

        for weather_condition, weather_dict in metrics_dict.items():

            # calculate distances to ideal points
            for vt, metrics in weather_dict[max_auc_weather[weather_condition][1]].items():
                tpr = metrics.recall
                fpr = 1 - metrics.specificity
                precision = metrics.precision

                roc_dist = distance.euclidean(
                    [fpr, tpr], roc_ideal)
                pr_dist = distance.euclidean(
                    [tpr, precision], pr_ideal)
                distances_roc.append([roc_dist, vt])
                distances_pr.append([pr_dist, vt])

            # sort distances by ROC and PR distance, and select the two best vacant thresholds
            sorted_distances_roc = np.array(
                sorted(distances_roc, key=lambda x: x[0]))
            sorted_distances_pr = np.array(
                sorted(distances_pr, key=lambda x: x[0]))

            # print(sorted_distances_roc)
            # print(sorted_distances_pr)

            distance_roc = sorted_distances_roc[0][0]
            distance_pr = sorted_distances_pr[0][0]
            best_vt_roc = sorted_distances_roc[0][1]
            best_vt_pr = sorted_distances_pr[0][1]
            best_metric_roc = weather_dict[max_auc_weather[weather_condition]
                                           [1]][best_vt_roc]
            best_metric_pr = weather_dict[max_auc_weather[weather_condition][1]][best_vt_pr]

            best_metrics.append([weather_condition, max_auc_weather[weather_condition][1], best_vt_roc,  max_auc_weather[weather_condition][0],
                                "ROC", distance_roc, best_metric_roc.precision, best_metric_roc.specificity, best_metric_roc.recall, best_metric_roc.f1, best_metric_roc.accuracy])

            best_metrics.append([weather_condition, max_auc_weather[weather_condition][1], best_vt_pr,  max_auc_weather[weather_condition][0],
                                "PR", distance_pr, best_metric_pr.precision, best_metric_pr.specificity, best_metric_pr.recall, best_metric_pr.f1, best_metric_pr.accuracy])
            print(best_metric_roc.to_latex("Training",
                                           weather_condition, max_auc_weather[weather_condition][1], best_vt_roc))
            print(best_metric_pr.to_latex("Training",
                                          weather_condition, max_auc_weather[weather_condition][1], best_vt_pr))
        df = pd.DataFrame(best_metrics, columns=[
                          "Weather", first_param_str, "vt", "AUC", "Dist Curve", "Dist", "Precision", "Specificity", "Recall", "F1", "Accuracy"])
        return df


def scatter_metrics(metrics_list):
    """
    Plots the ROC and PR curves for the given list of PerformanceMetrics objects.
    """

    fontP = FontProperties(family='sans-serif', size=8)
    # fontP.set_size('xx-small')

    fig, ax = plt.subplots(2, 1, figsize=(5, 10), dpi=300, sharey=True)

    for metrics in metrics_list:
        tpr = metrics.recall
        fpr = 1 - metrics.specificity
        precision = metrics.precision

        ax[0].scatter(tpr, fpr, marker='o')
        ax[1].scatter(precision, tpr, marker='o')

    # # Optional: plot extra metrics
    # if metrics_extra is not None:
    #     for label, metrics in metrics_extra.items():
    #         tpr = metrics.recall
    #         fpr = 1 - metrics.specificity
    #         precision = metrics.precision

    #         # ROC CURVE
    #         # li = zip(*[fpr, tpr])
    #         ax[0].plot((fpr, tpr), marker='o',
    #                    label=f'{index}: {label}')

    #         # PR CURVE
    #         # li_recall = zip(*[tpr, precision])
    #         ax[1].plot((tpr, precision), linestyle='--', marker='o',
    #                    label=f'{index}: {label}')

    #         index += 1

    # Set fixed x and y axis limits and ticks
    for axi in ax:
        axi.set_xlim([-0.025, 1.025])
        axi.set_ylim([-0.025, 1.025])
        axi.set_xticks(np.arange(0, 1.1, 0.1))
        axi.set_yticks(np.arange(0, 1.05, 0.05))
        axi.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        axi.yaxis.set_major_locator(plt.MultipleLocator(0.05))
        axi.grid(color='gray', linestyle='--', linewidth=0.5)
        axi.legend(loc='lower right', prop=fontP)
        axi.tick_params(axis='both', labelsize=6)

    # Graph 1: FPR-TPR
    plt.subplot(2, 1, 1)
    # fig.suptitle(f'{parking_id}-{weather}')
    ax[0].set_title('ROC Curve')
    ax[0].set_xlabel('False positive rate (1 - Specificity)', fontsize=8)
    ax[0].set_ylabel('True positive rate (Recall)', fontsize=8)

    plt.plot(0, 1, marker='x',
             label='Perfect classifier')

    # Graph 2: Recall-TPR
    plt.subplot(2, 1, 2)
    ax[1].set_title('PR Curve')
    ax[1].set_xlabel('True positive rate (Recall)', fontsize=8)
    ax[1].set_ylabel('Precision', fontsize=8)

    plt.plot(1, 1, marker='x',
             label='Perfect classifier')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=1,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.2)

    # if save_fig_dst is not None:
    #     plt.savefig(save_fig_dst, facecolor='white', bbox_inches='tight')

    plt.show()
