class PerformanceMetrics:

    def __init__(self, real: bool(), predicted: bool(), precision: float = None, specificity: float = None, recall: float = None, f1: float = None, accuracy: float = None):
        self.real = real
        self.predicted = predicted
        self.precision = precision
        self.specificity = specificity
        self.recall = recall
        self.f1 = f1
        self.accuracy = accuracy
