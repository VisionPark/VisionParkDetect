from src.data.entity.Space import Space
from src.detector.entity.DetectionParams import DetectionParams
from src.app.ParkingOccupancyProcessor import ParkingOccupancyProcessor
from src.data.ParkingProviderSqlite import ParkingProviderSqlite, ParkingProviderSqliteParams
from sys import path
import cv2 as cv
import cvzone
from src.metrics.PerformanceMetricsProvider import PerformanceMetricsProvider
path.append("../")


class ParkingOccupancyProcessorSqlite(ParkingOccupancyProcessor):

    def __init__(self, parking_provider_params: ParkingProviderSqliteParams, detection_params: DetectionParams, performance_metrics_provider: PerformanceMetricsProvider):
        super().__init__(parking_provider_params,
                         detection_params, performance_metrics_provider)

        self.parking_provider: ParkingProviderSqlite = ParkingProviderSqlite(
            parking_provider_params)

    def getDetectionImg(self, parking_img: cv.Mat, parking_spaces: list[Space], real, predicted):
        img = parking_img
        for i, space in enumerate(parking_spaces):

            if predicted[i] == 1:
                cv.polylines(img, [space.vertex], True,
                             (0, 255, 0), thickness=2)
            elif predicted[i] == 0:
                cv.polylines(img, [space.vertex], True,
                             (0, 0, 255), thickness=2)

            # Print detection value
            if(space.count is not None):
                v = space.vertex.reshape(-1, 1, 2)
                cols = v[:, :, 0].flatten()
                rows = v[:, :, 1].flatten()

                row_max = max(rows)
                col_min = min(cols)

                text = str(
                    round(space.count/space.area, 2))
                cvzone.putTextRect(img, text, (col_min, row_max-2),
                                   scale=0.8, thickness=1, offset=0)
        return img
