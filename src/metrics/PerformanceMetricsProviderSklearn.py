import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn import metrics
from src.metrics.PerformanceMetricsProvider import PerformanceMetricsProvider
import pandas as pd

from src.metrics.entity.PerformanceMetrics import PerformanceMetrics

from sklearn.metrics import roc_auc_score, auc, average_precision_score
from adjustText import adjust_text
from matplotlib.font_manager import FontProperties
from numpy import arange, mean

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
            self.real, self.predicted).ravel()

        # Precision (or Sensivity) Score = TP / (FP + TP). Minimize FP
        precision = tp / (fp+tp)

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

        if(plot):
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
    def show_tpr_fpr(metrics_dict_vt: dict, parking_id, weather, show_diff=False, save_fig_dst=None):

        fontP = FontProperties()
        fontP.set_size('xx-small')

        fig, ax = plt.subplots(2, 1, figsize=(5, 10), dpi=300, sharey=True)

        # plt.figure()
        index = 0
        for vt, metrics_dict_diff in metrics_dict_vt.items():
            tpr_list = list()
            fpr_list = list()
            precision_list = list()
            diff_list = list()
            real_list = list()
            predicted_list = list()
            for diff, metrics in metrics_dict_diff.items():
                tpr_list.append(metrics.recall)
                fpr_list.append(1 - metrics.specificity)
                precision_list.append(metrics.precision)
                diff_list.append(diff)
                real_list = real_list + metrics.real
                predicted_list = predicted_list + metrics.predicted

            auc = round(roc_auc_score(real_list, predicted_list), 3)
            ap = round(average_precision_score(real_list, predicted_list), 3)

            # ROC CURVE
            li = zip(*[fpr_list, tpr_list])
            ax[0].plot(*zip(*li), linestyle='--', marker='o',
                       label=f'{index}: bs={vt} AUC={auc}')

            # PR CURVE
            li_recall = zip(*[tpr_list, precision_list])
            ax[1].plot(*zip(*li_recall), linestyle='--', marker='o',
                       label=f'{index}: bs={vt} AP={ap}')

            index += 1
        # Set fixed x and y axis limits and ticks
        for axi in ax:
            axi.set_xlim([-0.025, 1.025])
            axi.set_ylim([-0.025, 1.025])
            axi.set_xticks(arange(0, 1.1, 0.05))
            axi.set_yticks(arange(0, 1.1, 0.05))

        # Graph 1: FPR-TPR
        plt.subplot(2, 1, 1)
        fig.suptitle(f'{parking_id}-{weather}')
        ax[0].set_xlabel('False positive rate (1 - Specificity)', fontsize=8)
        ax[0].set_ylabel('True positive rate (Recall)', fontsize=8)
        # plt.xticks(arange(0, 1.05, 0.05), fontsize=6)
        # plt.yticks(arange(min(0.5, min(tpr_list)), 1.05, 0.05), fontsize=6)
        ax[0].locator_params(axis='both', tight=True, nbins=15)
        ax[0].tick_params(axis='both', labelsize=6)

        ax[0].plot(0, 1, marker='x',
                   label='Perfect classifier')
        plt.legend(loc='lower right', prop=fontP)

        if show_diff:
            texts = [plt.text(fpr_list[i], tpr_list[i], diff_list[i], size=8)
                     for i in range(len(fpr_list))]
            adjust_text(texts, arrowprops={
                        'arrowstyle': 'fancy'}, expand_points=(1.3, 1.3))

        # Graph 2: Recall-TPR
        plt.subplot(2, 1, 2)
        ax[1].set_xlabel('True positive rate (Recall)', fontsize=8)
        ax[1].set_ylabel('Precision', fontsize=8)
        ax[1].locator_params(axis='both', tight=True, nbins=15)
        ax[1].tick_params(axis='both', labelsize=6)
        # plt.xticks(arange(min(0.5, min(tpr_list)), 1.05, 0.05), fontsize=6)
        # plt.yticks(arange(min(0.5, min(precision_list)), 1.05, 0.05), fontsize=6)

        # if len(ax[1].get_xticks()) < 5:
        #     step = (1-min(tpr_list))/10
        #     ax[1].set_xticks(
        #         arange(min(tpr_list), 1+step, step))

        ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        # We change the fontsize of minor ticks label

        plt.plot(1, 1, marker='x',
                 label='Perfect classifier')
        ax[1].legend(loc='lower right', prop=fontP)

        if show_diff:
            texts = [plt.text(tpr_list[i], precision_list[i], diff_list[i], size=8)
                     for i in range(len(fpr_list))]
            adjust_text(texts, arrowprops={
                        'arrowstyle': 'fancy'}, expand_points=(1.3, 1.3))

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=1,
                            top=0.9,
                            wspace=0.1,
                            hspace=0.4)

        if save_fig_dst is not None:
            plt.savefig(save_fig_dst, facecolor='white', bbox_inches='tight')

        plt.show()

    @staticmethod
    def choose_parameters(metrics_dict: dict):
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

        bs_auc_dict = dict()
        bs_ap_dict = dict()

        max_auc_global = 0
        max_auc_global_bs = 0
        max_auc_weather = ""
        max_f1_global = 0
        max_f1_global_vt = 0

        for weather_condition, weather_dict in metrics_dict.items():

            max_auc = 0
            max_auc_bs = 0
            max_f1 = 0
            max_f1_vt = 0

            for bs, vacant_dict in weather_dict.items():
                real_list = list()
                predicted_list = list()
                for vt, metrics in vacant_dict.items():

                    real_list = real_list + metrics.real
                    predicted_list = predicted_list + metrics.predicted

                    bs_auc_dict[bs] = (roc_auc_score(
                        real_list, predicted_list))
                    bs_ap_dict[bs] = average_precision_score(
                        real_list, predicted_list)

                    if bs_auc_dict[bs] > max_auc:
                        max_auc = bs_auc_dict[bs]
                        max_auc_bs = bs

                if max_auc > max_auc_global:
                    max_auc_global = max_auc
                    max_auc_global_bs = max_auc_bs
                    max_auc_weather = weather_condition

            for vt, metrics in weather_dict[max_auc_bs].items():
                if metrics.f1 > max_f1:
                    max_f1 = metrics.f1
                    max_f1_vt = vt

            print(f"\nBest parameters for weather: {weather_condition}")
            print(
                f"block_size: {max_auc_bs}\t\t\tSelected from max AUC={max_auc}")
            print(
                f"vacant_threshold: {max_f1_vt}\t\tSelected from max f1={max_f1}")

        for vt, metrics in metrics_dict[max_auc_weather][max_auc_global_bs].items():
            if metrics.f1 > max_f1_global:
                max_f1_global = metrics.f1
                max_f1_global_vt = vt

        print("\nBest GLOBAL parameters")
        print(
            f"block_size: {max_auc_global_bs}\t\t\tSelected from max AUC={max_auc_global} and weather {max_auc_weather}")
        print(
            f"vacant_threshold: {max_f1_global_vt}\t\tSelected from max f1={max_f1_global}")

        weather_condition_list = list()
        vacant_threshold_list = list()
        block_size_list = list()
        auc_list = list()
        ap_list = list()
        precision_list = list()
        specificity_list = list()
        recall_list = list()
        f1_list = list()
        accuracy_list = list()

        # Loop over weather conditions
        for weather_condition, weather_dict in metrics_dict.items():
            # Initialize lists to store data for each weather condition
            auc_weather = []
            ap_weather = []
            precision_weather = []
            specificity_weather = []
            recall_weather = []
            f1_weather = []
            accuracy_weather = []

            bs_auc_dict = dict()
            bs_ap_dict = dict()

            for bs, vacant_dict in weather_dict.items():
                real_list = list()
                predicted_list = list()
                for vt, metrics in vacant_dict.items():

                    real_list = real_list + metrics.real
                    predicted_list = predicted_list + metrics.predicted

                bs_auc_dict[bs] = (roc_auc_score(real_list, predicted_list))
                bs_ap_dict[bs] = average_precision_score(
                    real_list, predicted_list)

            for block_size, vacant_dict in weather_dict.items():
                for vacant_threshold, perf_metrics in vacant_dict.items():
                    # Append performance metrics for each curve to the respective lists
                    auc_weather.append(bs_auc_dict[block_size])
                    ap_weather.append(bs_ap_dict[block_size])
                    precision_weather.append(perf_metrics.precision)
                    specificity_weather.append(perf_metrics.specificity)
                    recall_weather.append(perf_metrics.recall)
                    f1_weather.append(perf_metrics.f1)
                    accuracy_weather.append(perf_metrics.accuracy)
                    vacant_threshold_list.append(vacant_threshold)
                    block_size_list.append(block_size)
                    weather_condition_list.append(weather_condition)

            # Append average performance metrics for the weather condition to the respective lists
            auc_list = auc_list + auc_weather
            ap_list = ap_list + ap_weather
            precision_list = precision_list + precision_weather
            specificity_list = specificity_list + specificity_weather
            recall_list = recall_list + recall_weather
            f1_list = f1_list + f1_weather
            accuracy_list = accuracy_list + accuracy_weather

        # Create a pandas dataframe to store the metrics
        metrics_df = pd.DataFrame({
            "weather_condition": weather_condition_list,
            "block_size": block_size_list,
            "vacant_threshold": vacant_threshold_list,
            "AUC": auc_list,
            "AP": ap_list,
            "precision": precision_list,
            "specificity": specificity_list,
            "recall": recall_list,
            "F1": f1_list,
            "accuracy": accuracy_list
        })

        return metrics_df

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
