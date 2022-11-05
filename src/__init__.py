from src.data.ParkingProvider import NoSpacesException, NoImageException
from src.metrics.PerformanceMetricsProviderSklearn import PerformanceMetricsProviderSklearn
from src.metrics.entity.PerformanceMetrics import PerformanceMetrics
from src.detector.entity.DetectionParams import DetectionParams
from src.data.ParkingProviderLocal import ParkingProviderLocalParams
from src.app.ParkingOccupancyProcessorLocal import ParkingOccupancyProcessorLocal
import cv2 as cv
import sys
import argparse
sys.path.insert(0, './src')


def main():
    print("Hello World!")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('iterations', type=int, nargs='?', const='-1',
                        help='Number of iterations to process. By default processing until no space or image available.')

    it = parser.parse_args().iterations

    PARAMS_UFPR04 = DetectionParams(
        (3, 3), 0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 41, 11, 3, 104, vacant_threshold=0.25, show_imshow=False)  # UFPR04

    parking_id = "UFPR04"
    provider_params = ParkingProviderLocalParams(
        parking_id=parking_id, path='E:\\Documents\\PKLot\\PKLot\\PKLot\\PKLot\\'+parking_id, k=30)

    metrics: PerformanceMetricsProviderSklearn = PerformanceMetricsProviderSklearn()

    processor = ParkingOccupancyProcessorLocal(
        parking_provider_params=provider_params, detection_params=PARAMS_UFPR04, performance_metrics_provider=metrics)

    try:
        i = 0
        # while (it > 0 and i < it) or it < 0:
        #     i = i+1
        #     print(f"Processing sample: {str(i)}")
        #     processor.process()

        processor.process_batch()

    except (NoSpacesException, NoImageException) as ex:
        print(f"Finished processing {str(i)} samples: ", ex)

    metrics.calculate_metrics()
    metrics.show_confusion_matrix()

    print("Bye World!")


if __name__ == "__main__":
    main()
