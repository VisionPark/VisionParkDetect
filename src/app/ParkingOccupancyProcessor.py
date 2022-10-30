from abc import ABC
from src.detector.entity.DetectionParams import DetectionParams
from src.detector.OccupancyDetectorBorders import OccupancyDetectorBorders
from src.detector.OcupancyDetector import OccupancyDetector
from src.metrics.PerformanceMetricsProvider import PerformanceMetricsProvider
from src.metrics.PerformanceMetricsProviderSklearn import PerformanceMetricsProviderSklearn
from src.metrics.entity.PerformanceMetrics import PerformanceMetrics
from src.data.ParkingProvider import ParkingProvider
from abc import ABC


class ParkingOccupancyProcessor(ABC):
    def __init__(self, parking_provider_params: ParkingProvider.ParkingProviderParams, detection_params: DetectionParams):

        # Overriden by child implementation
        self.parking_provider = ParkingProvider(parking_provider_params)

        self.occupancy_detector: OccupancyDetector = OccupancyDetectorBorders(
            detection_params)

        self.performance_metrics: PerformanceMetricsProvider = PerformanceMetricsProviderSklearn()

    def process(self) -> PerformanceMetrics:

        parking = self.parking_provider.get_parking()

        self.occupancy_detector.detect_image(
            parking.image, parking.image_date, parking.spaces)

        self.performance_metrics = PerformanceMetricsProviderSklearn(
            parking.spaces).calculate_metrics()

        return self.performance_metrics
