from src.metrics.PerformanceMetricsProviderSklearn import PerformanceMetricsProviderSklearn
from src.metrics.entity.PerformanceMetrics import PerformanceMetrics
from src.detector.entity.DetectionParams import DetectionParams
from src.data.ParkingProviderLocal import ParkingProviderLocalParams
from src.app.ParkingOccupancyProcessorLocal import ParkingOccupancyProcessorLocal
import cv2 as cv
import sys
sys.path.insert(0, './src')


sys.path.insert(0, './src')


def main():
    print("Hello World!")
    PARAMS_UFPR04 = DetectionParams(
        (3, 3), 0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 41, 11, 3, 104, vacant_threshold=0.25, show_imshow=False)  # UFPR04

    parking_id = "UFPR04"
    provider_params = ParkingProviderLocalParams(
        parking_id=parking_id, path='E:\\Documents\\PKLot\\PKLot\\PKLot\\PKLot\\'+parking_id, k=4)

    metrics: PerformanceMetricsProviderSklearn = PerformanceMetricsProviderSklearn()

    processor = ParkingOccupancyProcessorLocal(
        parking_provider_params=provider_params, detection_params=PARAMS_UFPR04, performance_metrics_provider=metrics)

    for i in range(20):
        print(f"Processing: {str(i)}")
        processor.process()

    metrics.calculate_metrics()
    metrics.show_confusion_matrix()

    print("Bye World!")


if __name__ == "__main__":
    main()
