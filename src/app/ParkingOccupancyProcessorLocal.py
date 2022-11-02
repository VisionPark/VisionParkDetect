from src.data.ParkingProviderLocal import ParkingProviderLocal, ParkingProviderLocalParams
from src.app.ParkingOccupancyProcessor import ParkingOccupancyProcessor
from src.detector.entity.DetectionParams import DetectionParams
from sys import path

from src.metrics.PerformanceMetricsProvider import PerformanceMetricsProvider
path.append("../../")


class ParkingOccupancyProcessorLocal(ParkingOccupancyProcessor):

    def __init__(self, parking_provider_params: ParkingProviderLocalParams, detection_params: DetectionParams, performance_metrics_provider: PerformanceMetricsProvider):
        super().__init__(parking_provider_params,
                         detection_params, performance_metrics_provider)

        self.parking_provider: ParkingProviderLocal = ParkingProviderLocal(
            parking_provider_params)
