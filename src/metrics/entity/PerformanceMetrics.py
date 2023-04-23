class PerformanceMetrics:

    def __init__(self, real: bool(), predicted: bool(), precision: float = None, specificity: float = None, recall: float = None, f1: float = None, accuracy: float = None):
        self.real = real
        self.predicted = predicted
        self.precision = precision
        self.specificity = specificity
        self.recall = recall
        self.f1 = f1
        self.accuracy = accuracy

    def to_latex(self, type=None, weather=None, bs=None, vt=None):
        return f"{type} & {weather} & {bs} & {vt} & {self.precision:.3f} & {self.specificity:.3f} & {self.recall:.3f} & {self.f1:.3f} & {self.accuracy:.3f}"
