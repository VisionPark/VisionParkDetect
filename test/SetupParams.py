from src.data.ParkingProvider import NoSpacesException, NoImageException
from src.detector.OccupancyDetectorBorders import OccupancyDetectorBorders
from src.metrics.PerformanceMetricsProviderSklearn import PerformanceMetricsProviderSklearn
from src.metrics.entity.PerformanceMetrics import PerformanceMetrics
from src.detector.entity.DetectionParams import DetectionParams
from src.data.ParkingProviderLocal import ParkingProviderLocalParams, ParkingProviderLocal
from src.app.ParkingOccupancyProcessorLocal import ParkingOccupancyProcessorLocal
import cv2 as cv
import argparse


def main():
    print("Hello World!")

    parser = argparse.ArgumentParser(description='Setup detector parameters')
    parser.add_argument('parking', type=str, nargs='?', default='UFPR04', const='UFPR04',
                        help='Parking id to setup [UFPR04, UFPR05 or PUCPR]')

    PARAMS_UFPR04 = DetectionParams(
        (3, 3), 0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 41, 11, 3, 104, vacant_threshold=0.25, show_imshow=False)  # UFPR04

    parking_id = parser.parse_args().parking
    provider_params = ParkingProviderLocalParams(
        parking_id=parking_id, path='E:\\Documents\\PKLot\\PKLot\\PKLot\\PKLot\\'+parking_id, k=30)

    parking_provider = ParkingProviderLocal(provider_params)

    # metrics: PerformanceMetricsProviderSklearn = PerformanceMetricsProviderSklearn()

    # processor = ParkingOccupancyProcessorLocal(
    #     parking_provider_params=provider_params, detection_params=PARAMS_UFPR04, performance_metrics_provider=metrics)

    try:
        i = 0
        # while (it > 0 and i < it) or it < 0:
        #     i = i+1
        #     print(f"Processing sample: {str(i)}")
        #     processor.process()

        [img, _] = parking_provider.fetch_image()
        spaces = parking_provider.fetch_spaces()

        OccupancyDetectorBorders.setup_params(img, spaces)

    except (NoSpacesException, NoImageException) as ex:
        print(f"Finished processing: ", ex)

    print("Bye World!")


if __name__ == "__main__":
    main()
