import matplotlib.pyplot as plt
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
# Recall nos da la cantidad: ¿qué porcentaje de la clase positiva hemos sido capaces de identificar?
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

        # Recall Score = TP / (FN + TP). Minimize FN
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
    def show_tpr_fpr(metrics_dict_vt: dict, show_diff=False):

        fontP = FontProperties()
        fontP.set_size('xx-small')

        plt.figure(figsize=(5, 5), dpi=300)
        index = 0
        for vt, metrics_dict_diff in metrics_dict_vt.items():
            tpr_list = list()
            fpr_list = list()
            recall_list = list()
            diff_list = list()
            for diff, metrics in metrics_dict_diff.items():
                tpr_list.append(metrics.precision)
                fpr_list.append(1 - metrics.specificity)
                recall_list.append(metrics.recall)
                diff_list.append(diff)

        # auc_1 = auc(fpr, tpr)

            plt.subplot(2, 1, 1)
            li = zip(*[fpr_list, tpr_list])
            plt.plot(*zip(*li), linestyle='--', marker='o',
                     label=f'{index}: vt={vt}')

            plt.subplot(2, 1, 2)
            li_recall = zip(*[recall_list, tpr_list])
            plt.plot(*zip(*li_recall), linestyle='--', marker='o',
                     label=f'{index}: vt={vt}')

            index += 1

        plt.subplot(2, 1, 1)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.xticks(arange(0, 1.05, 0.1))
        plt.yticks(arange(0.5, 1.05, 0.05))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
        plt.plot(xy=(0, 1), linestyle='--', marker='o',
                 label='Perfect classifier')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)

        for i, txt in enumerate(diff_list):
            plt.annotate(txt, (fpr_list[i], tpr_list[i]))

        plt.subplot(2, 1, 2)
        plt.xlabel('Recall')
        plt.ylabel('True positive rate')

        plt.xticks(arange(0, 1.05, 0.05))
        plt.yticks(arange(0.5, 1.05, 0.05))

        plt.plot(xy=(1, 1), linestyle='--', marker='o',
                 label='Perfect classifier')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)

        for i, txt in enumerate(diff_list):
            plt.annotate(txt, (recall_list[i], tpr_list[i]))

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=1,
                            top=0.9,
                            wspace=0.1,
                            hspace=0.4)
        plt.show()

    @staticmethod
    def show_precision_recall(metrics_list: list[PerformanceMetrics], row_names: list[str]):
        precision_list = [m.precision for m in metrics_list]
        recall_list = [m.recall for m in metrics_list]

        plt.figure(figsize=(5, 5), dpi=300)
        for f, p in zip(recall_list, precision_list):
            plt.plot(f, p, 'x')

        # plt.scatter(fpr, tpr)

        texts = [plt.text(recall_list[i], precision_list[i], str(i), size=8)
                 for i in range(len(precision_list))]
        adjust_text(texts, arrowprops={
                    'arrowstyle': 'fancy'}, expand_points=(1.3, 1.3))

        plt.xlabel('Recall')
        plt.ylabel('Precision')

        fontP = FontProperties()
        fontP.set_size('xx-small')

        plt.legend([str(i)+": " + row[row.find(" "):] for i, row in enumerate(
            row_names)], bbox_to_anchor=(1.1, 1.05), loc='upper left', prop=fontP)
        plt.show()
