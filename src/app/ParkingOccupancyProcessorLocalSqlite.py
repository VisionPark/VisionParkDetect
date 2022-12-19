from src.app.ParkingOccupancyProcessorLocal import ParkingOccupancyProcessorLocal
from src.data.ParkingProviderLocalSqlite import ParkingProviderLocalSqlite, ParkingProviderLocalSqliteParams
from src.data.entity.Parking import Parking
from src.detector.OccupancyDetectorBorders import OccupancyDetectorBorders
from src.detector.entity.DetectionParams import DetectionParams
from sys import path
from src.metrics.PerformanceMetricsProvider import PerformanceMetricsProvider
path.append("../../")

fakeSelf = None


class ParkingOccupancyProcessorLocalSqlite(ParkingOccupancyProcessorLocal):

    def __init__(self, parking_provider_params: ParkingProviderLocalSqliteParams, detection_params: DetectionParams, performance_metrics_provider: PerformanceMetricsProvider):
        super().__init__(parking_provider_params,
                         detection_params, performance_metrics_provider)

        self.parking_provider: ParkingProviderLocalSqlite = ParkingProviderLocalSqlite(
            parking_provider_params)
        
