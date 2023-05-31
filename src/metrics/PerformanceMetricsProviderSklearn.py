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
        all_tpr_list = list()
        all_fpr_list = list()
        all_precision_list = list()

        def process_diff(diff, metrics):
            tpr_list = metrics.recall
            fpr_list = 1 - metrics.specificity
            precision_list = metrics.precision
            real_list = metrics.real
            predicted_list = metrics.predicted
            auc = round(roc_auc_score(real_list, predicted_list), 3)
            ap = round(average_precision_score(real_list, predicted_list), 3)
            return tpr_list, fpr_list, precision_list, auc, ap, real_list, predicted_list

        def process_metrics_dict_diff(vt, metrics_dict_diff):
            tpr_list = list()
            fpr_list = list()
            precision_list = list()
            diff_list = list()
            real_list = list()
            predicted_list = list()

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = []
                for diff, metrics in metrics_dict_diff.items():
                    futures.append(executor.submit(
                        process_diff, diff, metrics))

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    tpr_list.append(result[0])
                    fpr_list.append(result[1])
                    precision_list.append(result[2])
                    # diff_list.append(diff)
                    real_list.extend(result[5])
                    predicted_list.extend(result[6])

                    all_tpr_list.append(result[0])
                    all_fpr_list.append(result[1])
                    all_precision_list.append(result[2])

                auc = round(roc_auc_score(real_list, predicted_list), 3)
                ap = round(average_precision_score(
                    real_list, predicted_list), 3)

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
                real_list = list()
                predicted_list = list()
                for vt, metrics in vacant_dict.items():

                    real_list = real_list + metrics.real
                    predicted_list = predicted_list + metrics.predicted

                auc = roc_auc_score(
                    real_list, predicted_list)
                ap = average_precision_score(
                    real_list, predicted_list)

                if auc > max_auc_weather[weather_condition][0]:
                    max_auc_weather[weather_condition][0] = auc
                    max_auc_weather[weather_condition][1] = bs

                # if auc > max_auc:
                #     max_auc = auc
                #     max_auc_bs = bs

        # for weather_condition, weather_dict in metrics_dict.items():
        #     real_list = list()
        #     predicted_list = list()
        #     for vt, metrics in weather_dict[max_auc_bs].items():

        #         real_list = real_list + metrics.real
        #         predicted_list = predicted_list + metrics.predicted

        #     auc = roc_auc_score(
        #         real_list, predicted_list)
        #     # ap = average_precision_score(
        #     #     real_list, predicted_list)

        #     max_auc_weather[weather_condition] = auc

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
        # if metrics.f1 > max_f1:
        #     max_f1 = metrics.f1
        #     max_f1_vt = vt
        #     max_f1_bs = max_auc_bs
        #     max_metrics = metrics

        # selected_metrics.append(max_metrics)

        # if max_f1 > max_f1_global:
        #     max_f1_global = max_f1
        #     max_f1_global_vt = max_f1_vt
        #     max_f1_global_bs = max_f1_bs
        #     max_f1_weather = weather_condition

        #     print(f"\nBest parameters for weather: {weather_condition}")
        #     print(
        #         f"{first_param_str}: {max_auc_bs}\t\t\tSelected from max AUC={max_auc}")
        #     print(
        #         f"vacant_threshold: {max_f1_vt}\t\tSelected from max f1={max_f1}")

        # print("\nBest GLOBAL parameters")
        # print(
        #     f"{first_param_str}: {max_f1_global_bs}\t\t\tSelected from max f1={max_f1_global} and weather {max_f1_weather}")
        # print(
        #     f"vacant_threshold: {max_f1_global_vt}\t\tSelected from max f1={max_f1_global}")

        # weather_condition_list = list()
        # vacant_threshold_list = list()
        # block_size_list = list()
        # auc_list = list()
        # ap_list = list()
        # precision_list = list()
        # specificity_list = list()
        # recall_list = list()
        # f1_list = list()
        # accuracy_list = list()

        # # Loop over weather conditions
        # for weather_condition, weather_dict in metrics_dict.items():
        #     # Initialize lists to store data for each weather condition
        #     auc_weather = []
        #     ap_weather = []
        #     precision_weather = []
        #     specificity_weather = []
        #     recall_weather = []
        #     f1_weather = []
        #     accuracy_weather = []

        #     bs_auc_dict = dict()
        #     bs_ap_dict = dict()

        #     for bs, vacant_dict in weather_dict.items():
        #         real_list = list()
        #         predicted_list = list()
        #         for vt, metrics in vacant_dict.items():

        #             real_list = real_list + metrics.real
        #             predicted_list = predicted_list + metrics.predicted

        #         bs_auc_dict[bs] = (roc_auc_score(real_list, predicted_list))
        #         bs_ap_dict[bs] = average_precision_score(
        #             real_list, predicted_list)

        #     for block_size, vacant_dict in weather_dict.items():
        #         for vacant_threshold, perf_metrics in vacant_dict.items():
        #             # Append performance metrics for each curve to the respective lists
        #             auc_weather.append(bs_auc_dict[block_size])
        #             ap_weather.append(bs_ap_dict[block_size])
        #             precision_weather.append(perf_metrics.precision)
        #             specificity_weather.append(perf_metrics.specificity)
        #             recall_weather.append(perf_metrics.recall)
        #             f1_weather.append(perf_metrics.f1)
        #             accuracy_weather.append(perf_metrics.accuracy)
        #             vacant_threshold_list.append(vacant_threshold)
        #             block_size_list.append(block_size)
        #             weather_condition_list.append(weather_condition)

        #     # Append average performance metrics for the weather condition to the respective lists
        #     auc_list = auc_list + auc_weather
        #     ap_list = ap_list + ap_weather
        #     precision_list = precision_list + precision_weather
        #     specificity_list = specificity_list + specificity_weather
        #     recall_list = recall_list + recall_weather
        #     f1_list = f1_list + f1_weather
        #     accuracy_list = accuracy_list + accuracy_weather

        # # Create a pandas dataframe to store the metrics
        # metrics_df = pd.DataFrame({
        #     "weather_condition": metrics_dict.keys(),
        #     "block_size": block_size_list,
        #     "vacant_threshold": vacant_threshold_list,
        #     "AUC": auc_list,
        #     "precision": precision_list,
        #     "specificity": specificity_list,
        #     "recall": recall_list,
        #     "F1": f1_list,
        #     "accuracy": accuracy_list
        # })

        # return metrics_df

    # @staticmethod
    # def choose_parameters(metrics_dict: dict, first_param_str="blockSize"):
    #     """
    #     Calculates the AUC and AP for all curves in the given nested dictionary, along with the average for each weather condition and for all conditions combined.

    #     Args:
    #     - metrics_dict: A nested dictionary containing performance metrics for different combinations of weather condition, blockSize and vacantTreshold.
    #                     The most outer dictionary has string keys for weather condition and the next inner dicitonary as keys.
    #                     The next outer dictionary has keys representing block_size values and inner dictionary as keys
    #                     representing vacant_threshold values. The values in the inner dictionary are instances of PerformanceMetrics.
    #     Returns:
    #     - A pandas dataframe containing the AUC, AP, and average metrics for each weather condition and for all conditions combined.
    #     """

    #     bs_auc_dict = dict()
    #     bs_ap_dict = dict()

    #     max_f1_global = 0
    #     max_f1_global_vt = 0
    #     max_f1_global_bs = 0

    #     for weather_condition, weather_dict in metrics_dict.items():

    #         max_auc = 0
    #         max_auc_bs = 0
    #         max_f1 = 0
    #         max_f1_vt = 0
    #         max_f1_bs = 0
    #         max_metrics = None

    #         for bs, vacant_dict in weather_dict.items():
    #             real_list = list()
    #             predicted_list = list()
    #             for vt, metrics in vacant_dict.items():

    #                 real_list = real_list + metrics.real
    #                 predicted_list = predicted_list + metrics.predicted

    #             bs_auc_dict[bs] = (roc_auc_score(
    #                 real_list, predicted_list))
    #             bs_ap_dict[bs] = average_precision_score(
    #                 real_list, predicted_list)

    #             if bs_auc_dict[bs] > max_auc:
    #                 max_auc = bs_auc_dict[bs]
    #                 max_auc_bs = bs

    #         for vt, metrics in weather_dict[max_auc_bs].items():
    #             if metrics.f1 > max_f1:
    #                 max_f1 = metrics.f1
    #                 max_f1_vt = vt
    #                 max_f1_bs = max_auc_bs
    #                 max_metrics = metrics

    #         print(max_metrics.to_latex("Training",
    #                                    weather_condition, max_auc_bs, max_f1_vt))

    #         if max_f1 > max_f1_global:
    #             max_f1_global = max_f1
    #             max_f1_global_vt = max_f1_vt
    #             max_f1_global_bs = max_f1_bs
    #             max_f1_weather = weather_condition

    #         print(f"\nBest parameters for weather: {weather_condition}")
    #         print(
    #             f"{first_param_str}: {max_auc_bs}\t\t\tSelected from max AUC={max_auc}")
    #         print(
    #             f"vacant_threshold: {max_f1_vt}\t\tSelected from max f1={max_f1}")

    #     print("\nBest GLOBAL parameters")
    #     print(
    #         f"{first_param_str}: {max_f1_global_bs}\t\t\tSelected from max f1={max_f1_global} and weather {max_f1_weather}")
    #     print(
    #         f"vacant_threshold: {max_f1_global_vt}\t\tSelected from max f1={max_f1_global}")

    #     weather_condition_list = list()
    #     vacant_threshold_list = list()
    #     block_size_list = list()
    #     auc_list = list()
    #     ap_list = list()
    #     precision_list = list()
    #     specificity_list = list()
    #     recall_list = list()
    #     f1_list = list()
    #     accuracy_list = list()

    #     # Loop over weather conditions
    #     for weather_condition, weather_dict in metrics_dict.items():
    #         # Initialize lists to store data for each weather condition
    #         auc_weather = []
    #         ap_weather = []
    #         precision_weather = []
    #         specificity_weather = []
    #         recall_weather = []
    #         f1_weather = []
    #         accuracy_weather = []

    #         bs_auc_dict = dict()
    #         bs_ap_dict = dict()

    #         for bs, vacant_dict in weather_dict.items():
    #             real_list = list()
    #             predicted_list = list()
    #             for vt, metrics in vacant_dict.items():

    #                 real_list = real_list + metrics.real
    #                 predicted_list = predicted_list + metrics.predicted

    #             bs_auc_dict[bs] = (roc_auc_score(real_list, predicted_list))
    #             bs_ap_dict[bs] = average_precision_score(
    #                 real_list, predicted_list)

    #         for block_size, vacant_dict in weather_dict.items():
    #             for vacant_threshold, perf_metrics in vacant_dict.items():
    #                 # Append performance metrics for each curve to the respective lists
    #                 auc_weather.append(bs_auc_dict[block_size])
    #                 ap_weather.append(bs_ap_dict[block_size])
    #                 precision_weather.append(perf_metrics.precision)
    #                 specificity_weather.append(perf_metrics.specificity)
    #                 recall_weather.append(perf_metrics.recall)
    #                 f1_weather.append(perf_metrics.f1)
    #                 accuracy_weather.append(perf_metrics.accuracy)
    #                 vacant_threshold_list.append(vacant_threshold)
    #                 block_size_list.append(block_size)
    #                 weather_condition_list.append(weather_condition)

    #         # Append average performance metrics for the weather condition to the respective lists
    #         auc_list = auc_list + auc_weather
    #         ap_list = ap_list + ap_weather
    #         precision_list = precision_list + precision_weather
    #         specificity_list = specificity_list + specificity_weather
    #         recall_list = recall_list + recall_weather
    #         f1_list = f1_list + f1_weather
    #         accuracy_list = accuracy_list + accuracy_weather

    #     # Create a pandas dataframe to store the metrics
    #     metrics_df = pd.DataFrame({
    #         "weather_condition": weather_condition_list,
    #         "block_size": block_size_list,
    #         "vacant_threshold": vacant_threshold_list,
    #         "AUC": auc_list,
    #         "AP": ap_list,
    #         "precision": precision_list,
    #         "specificity": specificity_list,
    #         "recall": recall_list,
    #         "F1": f1_list,
    #         "accuracy": accuracy_list
    #     })

    #     return metrics_df

        # # Find block_size with the highest average AUC across all weather conditions
        # highest_auc_threshold = metrics_df.groupby(
        #     "block_size")["AUC"].max().idxmax()

        # highest_auc_threshold_df = metrics_df[metrics_df['block_size']
        #                                       == highest_auc_threshold]

        # highest_auc_threshold_value = highest_auc_threshold_df.iloc[0]["AUC"]

        # # Find vacant_threshold with the highest sum of precision and recall across all weather conditions for the chosen block_size
        # # vacant_threshold_df = metrics_df.loc[metrics_df["block_size"] == highest_auc_threshold].groupby(
        # #     "vacant_threshold")[["precision", "recall"]].sum()
        # # highest_pr_sum_block_size = highest_auc_threshold_df["precision"] + \
        # #     highest_auc_threshold_df["recall"]
        # # highest_pr_sum_vacant_threshold_idmax = highest_pr_sum_block_size.idxmax()
        # # highest_pr_sum_vacant_threshold = highest_auc_threshold_df.loc[
        # #     highest_pr_sum_vacant_threshold_idmax]["vacant_threshold"]
        # # highest_pr_sum_vacant_threshold_value = highest_pr_sum_block_size.loc[
        # #     highest_pr_sum_vacant_threshold_idmax]

        # # Find vacant_threshold with the highest F1 score across all weather conditions for the chosen block_size
        # vacant_threshold_highest_f1 = highest_auc_threshold_df.loc[highest_auc_threshold_df["F1"].idxmax(
        # )]

        # vacant_threshold_df = metrics_df[metrics_df['block_size']
        #                                  == highest_auc_threshold]

        # highest_auc_threshold_value = vacant_threshold_df.iloc[0]["F1"]

        # # mean_precision = metrics_df.loc[(metrics_df['block_size'] == highest_auc_threshold) &
        # #                                 (metrics_df['vacant_threshold'] == highest_pr_sum_vacant_threshold)]['precision'].mean()

        # # Display the vacant_threshold and block_size with the highest average AUC and precision, respectively
        # print(f"\nBest GLOBAL parameters:")
        # print(
        #     f"block_size:  {highest_auc_threshold}\t\t\tHighest AUC of {round(highest_auc_threshold_value,3)}")
        # print(
        #     f"vacant_threshold: {vacant_threshold_highest_f1}\t\t")

        # # Find the best parameters for each weather condition
        # best_params = {}
        # for weather_condition in set(weather_condition_list):
        #     metrics_subset = metrics_df[metrics_df["weather_condition"]
        #                                 == weather_condition]

        #     # Find block_size with the highest AUC for the current weather condition
        #     highest_auc_threshold = metrics_subset.groupby("block_size")[
        #         "AUC"].mean().idxmax()

        #     # Find vacant_threshold with the highest precision+recall for the chosen block_size and weather condition
        #     vacant_threshold_df = metrics_subset.loc[metrics_subset["block_size"] == highest_auc_threshold].groupby(
        #         "vacant_threshold")[["precision", "recall"]].sum()
        #     highest_pr_sum_block_size = vacant_threshold_df["precision"] + \
        #         vacant_threshold_df["recall"]
        #     highest_pr_sum_vacant_threshold = highest_pr_sum_block_size.idxmax()

        #     best_params[weather_condition] = {
        #         "block_size": highest_auc_threshold,
        #         "vacant_threshold": highest_pr_sum_vacant_threshold,
        #         "AUC": metrics_subset.loc[(metrics_subset["block_size"] == highest_auc_threshold) & (metrics_subset["vacant_threshold"] == highest_pr_sum_vacant_threshold)]["AUC"].values[0],
        #         "precision": metrics_subset.loc[(metrics_subset["block_size"] == highest_auc_threshold) & (metrics_subset["vacant_threshold"] == highest_pr_sum_vacant_threshold)]["precision"].values[0],
        #         "recall": metrics_subset.loc[(metrics_subset["block_size"] == highest_auc_threshold) & (metrics_subset["vacant_threshold"] == highest_pr_sum_vacant_threshold)]["recall"].values[0]
        #     }

        # # Display the best parameters for each weather condition
        # for weather_condition in set(weather_condition_list):
        #     print(f"\nBest parameters for {weather_condition}:")
        #     print(
        #         f"block_size: {best_params[weather_condition]['block_size']}\t\t\tHighest AUC of {round(best_params[weather_condition]['AUC'],3)}")
        #     print(
        #         f"vacant_threshold: {best_params[weather_condition]['vacant_threshold']}\t\tHighest sum of PR for the chosen block_size ({round(best_params[weather_condition]['AUC'],3)}):{round(best_params[weather_condition]['precision']+best_params[weather_condition]['recall'],3)}")

        # return metrics_df

    # @staticmethod
    # def choose_parameters(metrics_dict: dict, first_param_str="blockSize"):
    #     bs_auc_dict = dict()
    #     bs_ap_dict = dict()
    #     roc_ideal = [0, 1]  # ideal point for ROC curve
    #     pr_ideal = [1, 1]  # ideal point for PR curve
    #     best_vts = []  # list to store the best vacant thresholds
    #     distances = []  # list to store the distances to the ideal points

    #     def process_weather_condition(weather_condition, weather_dict):
    #         nonlocal bs_auc_dict, bs_ap_dict, best_vts, distances

    #         max_auc = 0
    #         max_auc_bs = 0

    #         for bs, vacant_dict in weather_dict.items():
    #             real_list = list()
    #             predicted_list = list()
    #             for vt, metrics in vacant_dict.items():

    #                 real_list = real_list + metrics.real
    #                 predicted_list = predicted_list + metrics.predicted

    #             bs_auc_dict[bs] = (roc_auc_score(
    #                 real_list, predicted_list))
    #             bs_ap_dict[bs] = average_precision_score(
    #                 real_list, predicted_list)

    #             if bs_auc_dict[bs] > max_auc:
    #                 max_auc = bs_auc_dict[bs]
    #                 max_auc_bs = bs

    #         # calculate distances to ideal points
    #         tpr = bs_auc_dict[max_auc_bs].recall
    #         fpr = 1 - bs_auc_dict[max_auc_bs].specificity
    #         precision = bs_auc_dict[max_auc_bs].precision

    #         roc_dist = distance.euclidean([tpr, fpr], roc_ideal)
    #         pr_dist = distance.euclidean([precision, tpr], pr_ideal)
    #         distances.append([roc_dist, pr_dist, max_auc_bs, max_f1_vt])

    #     with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    #         futures = []
    #         for weather_condition, weather_dict in metrics_dict.items():
    #             futures.append(executor.submit(
    #                 process_weather_condition, weather_condition, weather_dict))

    #         for future in concurrent.futures.as_completed(futures):
    #             try:
    #                 result = future.result()
    #             except Exception as e:
    #                 print(f"Error: {e}")

    #     # sort distances by ROC and PR distance, and select the two best vacant thresholds
    #     sorted_distances = np.array(sorted(distances, key=lambda x: (x[0], x[1])))
    #     best_vts.append(sorted_distances[0, 3])
    #     best_vts.append(sorted_distances[1, 3])

    #     # create pandas dataframe with the selected vacant thresholds
    #     results_df = pd.DataFrame({'blockSize': [sorted_distances[0, 2], sorted_distances[1, 2]],
    #                             'vacant_threshold': best_vts,
    #                             'ROC_distance': [sorted_distances[0, 0], sorted_distances[1, 0]],
    #                             'PR_distance': [sorted_distances[0, 1], sorted_distances[1, 1]]})
    #     return results_df


def scatter_metrics(metrics_list):

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
