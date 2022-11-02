from abc import ABC, abstractmethod
from src.metrics.entity.PerformanceMetrics import PerformanceMetrics
from src.data.entity.Space import Space


class PerformanceMetricsProvider(ABC):
    def __init__(self, metrics: PerformanceMetrics = None):
        self.metrics = metrics

    def __init__(self, spaces: list[Space]):
        self.real, self.predicted = PerformanceMetricsProvider.get_real_predicted(
            spaces)
        self.metrics = PerformanceMetrics(self.real, self.predicted)

    def __init__(self):
        self.real = []
        self.predicted = []
        self.metrics = PerformanceMetrics(self.real, self.predicted)

    @staticmethod
    def get_real_predicted(spaces: list[Space]):
        real = []
        predicted = []

        for space in spaces:
            real.append(space.is_vacant_real)
            predicted.append(space.is_vacant)

        return real, predicted

    def add_real_predicted(self, real, predicted):
        self.real.extend(real)
        self.predicted.extend(predicted)

    @abstractmethod
    def calculate_metrics(self):
        pass

    @abstractmethod
    def show_confusion_matrix(self):
        pass
