from abc import ABC
from src.detector.entity.DetectionParams import DetectionParams
from src.detector.OccupancyDetectorBorders import OccupancyDetectorBorders
from src.detector.OcupancyDetector import OccupancyDetector
from src.metrics.PerformanceMetricsProvider import PerformanceMetricsProvider
from src.metrics.PerformanceMetricsProviderSklearn import PerformanceMetricsProviderSklearn
from src.metrics.entity.PerformanceMetrics import PerformanceMetrics
from src.data.ParkingProvider import ParkingProvider
from src.data.ParkingProvider import ParkingProviderParams
from sys import path
path.append("../")


class ParkingOccupancyProcessor(ABC):
    def __init__(self, parking_provider_params: ParkingProviderParams, detection_params: DetectionParams, performance_metrics_provider: PerformanceMetricsProvider):

        # Overriden by child implementation
        # self.parking_provider = ParkingProvider(parking_provider_params)

        self.occupancy_detector: OccupancyDetector = OccupancyDetectorBorders(
            detection_params)

        self.performance_metrics: PerformanceMetricsProvider = performance_metrics_provider

    def process(self) -> PerformanceMetricsProvider:

        parking = self.parking_provider.get_parking()

        self.occupancy_detector.detect_image(
            parking.image, parking.image_date, parking.spaces)

        real, predicted = PerformanceMetricsProviderSklearn.get_real_predicted(
            parking.spaces)

        self.performance_metrics.add_real_predicted(real, predicted)

        return self.performance_metrics
