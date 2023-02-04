import functools
import threading
import cv2 as cv
from src.data.ParkingProviderLocal import ParkingProviderLocal, ParkingProviderLocalParams
from src.app.ParkingOccupancyProcessor import ParkingOccupancyProcessor
from src.data.entity.Parking import Parking
from src.detector.OccupancyDetectorBorders import OccupancyDetectorBorders
from src.detector.OccupancyDetectorDiff import OccupancyDetectorDiff
from src.detector.entity.DetectionParams import DetectionParams
from sys import path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
import numpy as np
from src.metrics.PerformanceMetricsProvider import PerformanceMetricsProvider
from src.metrics.PerformanceMetricsProviderSklearn import PerformanceMetricsProviderSklearn
path.append("../../")

fakeSelf = None


class ParkingOccupancyProcessorLocal(ParkingOccupancyProcessor):

    def __init__(self, parking_provider_params: ParkingProviderLocalParams, detection_params: DetectionParams, performance_metrics_provider: PerformanceMetricsProvider):
        super().__init__(parking_provider_params,
                         detection_params, performance_metrics_provider)

        self.parking_provider: ParkingProviderLocal = ParkingProviderLocal(
            parking_provider_params)

    def after_detection_callback(self, progress, future):
        progress.update()
        real_part, predicted_part = future.result()
        self.performance_metrics.add_real_predicted(
            real_part, predicted_part)

    def process_batch(self, num_workers=multiprocessing.cpu_count()):
        num_files = self.parking_provider.get_num_files()
        print(f"Processing {num_files} files with {num_workers} workers")

        detection_params = self.occupancy_detector.params

        parkings = []
        for i in range(num_files):
            parkings.append(self.parking_provider.get_parking())

        with multiprocessing.Pool(num_workers) as pool:
            args = [(p, detection_params) for p in parkings]
            results = list(
                tqdm(pool.imap(detect_wrapper_star, args), total=len(args)))

        for r in results:
            self.performance_metrics.add_real_predicted(r[0], r[1])

        return self.performance_metrics


def detect_wrapper(parking: Parking, params: DetectionParams):

    print(str(parking.image_date))

    spaces = OccupancyDetectorDiff.detect_image(params,
                                                parking.image, parking.image_date, parking.spaces)

    real, predicted = PerformanceMetricsProviderSklearn.get_real_predicted(
        spaces)

    return real, predicted


def detect_wrapper_star(args):
    return detect_wrapper(*args)
