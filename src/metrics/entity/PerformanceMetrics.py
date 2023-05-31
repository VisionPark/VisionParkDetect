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

    def __str__(self):
        return str(f"[{len(self.predicted)} samples] F1:{self.f1:.2f} Accuracy:{self.accuracy:.2f})")

    def __repr__(self):
        return repr(str(self))
