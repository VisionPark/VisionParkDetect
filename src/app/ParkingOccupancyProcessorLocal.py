from src.data.ParkingProviderLocal import ParkingProviderLocal, ParkingProviderLocalParams
from src.app.ParkingOccupancyProcessor import ParkingOccupancyProcessor
from src.detector.entity.DetectionParams import DetectionParams
from sys import path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
from src.metrics.PerformanceMetricsProvider import PerformanceMetricsProvider
from src.metrics.PerformanceMetricsProviderSklearn import PerformanceMetricsProviderSklearn
path.append("../../")


class ParkingOccupancyProcessorLocal(ParkingOccupancyProcessor):

    def __init__(self, parking_provider_params: ParkingProviderLocalParams, detection_params: DetectionParams, performance_metrics_provider: PerformanceMetricsProvider):
        super().__init__(parking_provider_params,
                         detection_params, performance_metrics_provider)

        self.parking_provider: ParkingProviderLocal = ParkingProviderLocal(
            parking_provider_params)

    def detect_wrapper(self):
        parking = self.parking_provider.get_parking()
        self.occupancy_detector.detect_image(
            parking.image, parking.image_date, parking.spaces)
        real, predicted = PerformanceMetricsProviderSklearn.get_real_predicted(
            parking.spaces)
        #  self.performance_metrics.add_real_predicted(real, predicted)
        return real, predicted

    def process_batch(self, num_cores=multiprocessing.cpu_count()):
        num_files = self.parking_provider.get_num_files()
        print(f"Processing {num_files} files with {num_cores} cores")

        with ThreadPoolExecutor(max_workers=num_cores) as pool:
            with tqdm(total=num_files) as progress:
                futures = []

                for i in range(num_files):
                    future = pool.submit(self.detect_wrapper)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                for future in futures:
                    real_part, predicted_part = future.result()
                    self.performance_metrics.add_real_predicted(
                        real_part, predicted_part)

        return self.performance_metrics
