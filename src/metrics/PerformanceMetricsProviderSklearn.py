import matplotlib.pyplot as plt
from sklearn import metrics
from src.metrics.PerformanceMetricsProvider import PerformanceMetricsProvider


class PerformanceMetricsProviderSklearn(PerformanceMetricsProvider):

    def calculate_metrics(self):
        tn, fp, fn, tp = metrics.confusion_matrix(
            self.real, self.predicted).ravel()

        # Precision Score = TP / (FP + TP). Minimize FP
        precision = tp / (fp+tp)

        # Specificity score = TN / (TN+FP)
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

    def show_confusion_matrix(self):

        print('Precision: %.3f' % self.detection_result.stats.precision)
        print('specificity: %.3f' % self.detection_result.stats.specificity)
        print('Recall: %.3f' % self.detection_result.stats.recall)
        print('F1 Score: %.3f' % self.detection_result.stats.f1)
        print('Accuracy: %.3f' % self.detection_result.stats.accuracy)

        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=metrics.confusion_matrix(
                self.detection_result.real, self.detection_result.predicted),
            display_labels=['Occupied', 'Vacant'])
        cm_display.plot()
        plt.show()
