from src.detector.entity.DetectionParams import DetectionParams
from app.ParkingOccupancyProcessor import ParkingOccupancyProcessor
from src.data.ParkingProviderLocal import ParkingProviderLocal
from src.data.ParkingProviderSqlite import ParkingProviderSqlite


class ParkingOccupancyProcessorLocal(ParkingOccupancyProcessor):

    def __init__(self, parking_provider_params: ParkingProviderSqlite.ParkingProviderSqliteParams, detection_params: DetectionParams):
        super().__init__(parking_provider_params, detection_params)

        self.parking_provider: ParkingProviderSqlite = ParkingProviderSqlite(
            parking_provider_params)
