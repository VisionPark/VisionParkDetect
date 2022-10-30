from src.detector.entity.DetectionParams import DetectionParams
from src.app.ParkingOccupancyProcessor import ParkingOccupancyProcessor
from src.data.ParkingProviderLocal import ParkingProviderLocal


class ParkingOccupancyProcessorLocal(ParkingOccupancyProcessor):

    def __init__(self, parking_provider_params: ParkingProviderLocal.ParkingProviderLocalParams, detection_params: DetectionParams):
        super().__init__(parking_provider_params, detection_params)

        self.parking_provider: ParkingProviderLocal = ParkingProviderLocal(
            parking_provider_params)
