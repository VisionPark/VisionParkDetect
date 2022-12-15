from abc import ABC
from src.data.entity.Space import Space
from src.detector.entity.DetectionParams import DetectionParams
from src.detector.OccupancyDetectorBorders import OccupancyDetectorBorders
from src.detector.OcupancyDetector import OccupancyDetector
from src.metrics.PerformanceMetricsProvider import PerformanceMetricsProvider
from src.metrics.PerformanceMetricsProviderSklearn import PerformanceMetricsProviderSklearn
from src.metrics.entity.PerformanceMetrics import PerformanceMetrics
from src.data.ParkingProvider import ParkingProvider
from src.data.ParkingProvider import ParkingProviderParams
import cv2 as cv
import cvzone
from sys import path
path.append("../")


class ParkingOccupancyProcessor(ABC):
    def __init__(self, parking_provider_params: ParkingProviderParams, detection_params: DetectionParams, performance_metrics_provider: PerformanceMetricsProvider):

        # Overriden by child implementation
        self.parking_provider = None

        self.occupancy_detector: OccupancyDetector = OccupancyDetectorBorders(
            detection_params)

        self.performance_metrics: PerformanceMetricsProvider = performance_metrics_provider

    def getDetectionImg(self, parking_img: cv.Mat, parking_spaces: list[Space], real, predicted):
        img = parking_img
        for i, space in enumerate(parking_spaces):

            if predicted[i] == 1 and real[i] == 1:    # True positive
                cv.polylines(img, [space.vertex], True,
                             (0, 255, 0), thickness=2)
            elif predicted[i] == 0 and real[i] == 0:  # True negative
                cv.polylines(img, [space.vertex], True,
                             (0, 0, 255), thickness=2)
            elif predicted[i] == 0 and real[i] == 1:  # False negative
                cv.polylines(img, [space.vertex], True,
                             (58, 146, 255), thickness=2)
            else:                                 # False positive
                cv.polylines(img, [space.vertex], True,
                             (98, 169, 36), thickness=2)

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

    def process(self) -> PerformanceMetricsProvider:

        parking = self.parking_provider.get_parking()
        detection_params = self.occupancy_detector.params

        self.occupancy_detector.detect_image(detection_params,
                                             parking.image, parking.image_date, parking.spaces)

        real, predicted = PerformanceMetricsProviderSklearn.get_real_predicted(
            parking.spaces)

        self.performance_metrics.add_real_predicted(real, predicted)

        if self.occupancy_detector.params.show_imshow:
            img = self.getDetectionImg(
                parking.image, parking.spaces, real, predicted)

            cv.imshow("Parking Detection", img)
            key = cv.waitKey(0)

            if(key == 27):
                self.occupancy_detector.params.show_imshow = False
            cv.destroyAllWindows()

        return self.performance_metrics
