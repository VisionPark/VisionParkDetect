import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn import metrics
from src.metrics.PerformanceMetricsProvider import PerformanceMetricsProvider
import pandas as pd

from src.metrics.entity.PerformanceMetrics import PerformanceMetrics

from sklearn.metrics import roc_curve, auc
from adjustText import adjust_text
from matplotlib.font_manager import FontProperties
from numpy import arange

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

        fig, ax = plt.subplots(2, 1, figsize=(5, 5), dpi=300, sharey=True)

        # plt.figure()
        index = 0
        for vt, metrics_dict_diff in metrics_dict_vt.items():
            tpr_list = list()
            fpr_list = list()
            precision_list = list()
            diff_list = list()
            for diff, metrics in metrics_dict_diff.items():
                tpr_list.append(metrics.recall)
                fpr_list.append(1 - metrics.specificity)
                precision_list.append(metrics.precision)
                diff_list.append(diff)

            # auc_1 = round(auc(fpr_list, tpr_list), 2)
            # auc_2 = round(auc(precision_list, tpr_list), 2)
            # fig.sub
            # ax.subplot(2, 1, 1)
            li = zip(*[fpr_list, tpr_list])
            ax[0].plot(*zip(*li), linestyle='--', marker='o',
                       label=f'{index}: vt={vt}')

            # plt.subplot(2, 1, 2)
            li_recall = zip(*[tpr_list, precision_list])
            ax[1].plot(*zip(*li_recall), linestyle='--', marker='o',
                       label=f'{index}: vt={vt} auc=')

            index += 1

        # Graph 2: FPR-TPR
        plt.subplot(2, 1, 1)
        fig.suptitle(f'{parking_id}-{weather}')
        ax[0].set_xlabel('False positive rate (1 - Especificidad)', fontsize=8)
        ax[0].set_ylabel('True positive rate (Recall)', fontsize=8)
        # plt.xticks(arange(0, 1.05, 0.05), fontsize=6)
        # plt.yticks(arange(min(0.5, min(tpr_list)), 1.05, 0.05), fontsize=6)
        ax[0].locator_params(axis='both', tight=True, nbins=15)
        ax[0].tick_params(axis='both', labelsize=6)

        ax[0].plot(0, 1, marker='x',
                   label='Perfect classifier')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)

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

        ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # We change the fontsize of minor ticks label

        plt.plot(1, 1, marker='x',
                 label='Perfect classifier')
        ax[1].legend(bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)

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
