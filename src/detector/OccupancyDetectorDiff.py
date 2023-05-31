

from src.detector.entity.DetectionParams import DetectionParams
from src.data.entity.Space import Space
from src.detector.OcupancyDetector import OccupancyDetector
import cv2 as cv
from datetime import datetime
import numpy as np
import threading
import math
from skimage.exposure import match_histograms


class OccupancyDetectorDiff(OccupancyDetector):

    def __init__(self, params: DetectionParams):
        super().__init__(params)

    @staticmethod
    def detect_image(params: DetectionParams, parking_img: cv.Mat, parking_img_date: datetime, spaces: list[Space]):

        # imgPre = OccupancyDetectorDiff.preProcess(
        #     params, parking_img)
        # imgPre = OccupancyDetectorDiff.canny_edge_detection(
        #     parking_img, params, params.show_imshow)
        if params.match_histograms is not None and params.match_histograms:
            imgPre = OccupancyDetectorDiff.preProcess_match_histograms(
                params, parking_img)
        else:
            imgPre = OccupancyDetectorDiff.preProcess(
                params, parking_img)

        new_spaces = []
        for space in spaces.copy():
            vertex = space.vertex
            if (vertex.size == 0):
                continue
            vertex = vertex.reshape(4, 1, 2)

            # Get ROI of parking space
            roi = OccupancyDetector.get_roi(imgPre, vertex)

            # Decide if vacant depending on the detection area
            is_vacant = OccupancyDetectorDiff.is_space_vacant(
                roi, space, params.vacant_threshold)

            # Update space occupancy
            space.is_vacant = is_vacant
            space.since = parking_img_date
            new_spaces.append(space)

        return new_spaces

    @staticmethod
    def get_empty_img(parking_id, weather='Cloudy') -> cv.Mat:
        return cv.imread(
            f"E:/OneDrive - UNIVERSIDAD DE HUELVA/TFG/VisionParkDetect/dataset/empty/{parking_id}/{weather}/{parking_id}_{weather}_empty.jpg")

    @staticmethod
    def preProcess(params, img):
        img_empty = cv.cvtColor(OccupancyDetectorDiff.get_empty_img(
            params.parking_id, params.weather), cv.COLOR_BGR2GRAY)

        img_empty_blur = cv.GaussianBlur(img_empty, params.gb_k, params.gb_s)

        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_blur = cv.GaussianBlur(
            imgGray, params.gb_k, params.gb_s)

        diff = cv.absdiff(img_blur, img_empty_blur)
        imgThreshold = cv.threshold(
            diff, params.diff_threshold, 255, cv.THRESH_BINARY)[1]

        # Remove salt and pepper noise
        # if params.median_k != -1:
        #     imgMedian = cv.medianBlur(imgThreshold, params.median_k)
        # else:
        #     imgMedian = imgThreshold

        # Make thicker edges
        # kernel = np.ones((5,5), np.uint8)
        # imgEro = cv.erode(imgMedian, kernel, iterations=1)
        # imgDilate = cv.dilate(imgEro, kernel, iterations=1)

        # Remove small objects
        if params.bw_size != -1:
            imgBw = OccupancyDetectorDiff.bwareaopen(
                imgThreshold, params.bw_size)
        else:
            imgBw = imgThreshold
        # cv.imshow("IMG Dilate", imgDilate)

        if params.show_imshow:
            cv.imshow("1 - IMGBlur", img_blur)
            cv.imshow("2 - IMGEmptyBlur", img_empty_blur)
            cv.imshow("3 - diff_threshold", imgThreshold)
            cv.imshow("4 - imgBw", imgBw)

        return imgBw

    @staticmethod
    def preProcess_match_histograms(params, img):
        img_empty = OccupancyDetectorDiff.get_empty_img(
            params.parking_id, params.weather if params.weather is not None else 'Cloudy')

        img_matched = match_histograms(img, img_empty,
                                       multichannel=True)

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_empty_gray = cv.cvtColor(img_empty, cv.COLOR_BGR2GRAY)
        img_matched_gray = cv.cvtColor(img_matched, cv.COLOR_BGR2GRAY)

        diff = cv.absdiff(img_matched_gray, img_empty_gray)

        imgThreshold = cv.threshold(
            diff, params.diff_threshold, 255, cv.THRESH_BINARY)[1]

        if params.show_imshow:
            cv.imshow("1 - IMG", img)
            cv.imshow("1 - IMGEmpty", img_empty)
            cv.imshow("1 - img_empty_gray", img_empty_gray)
            cv.imshow('2 - Img matched', img_matched_gray)
            cv.imshow("3 - diff_threshold", imgThreshold)

            cv.imwrite('img_empty.png', img_empty)
            cv.imwrite('img_matched.png', img_matched)
            cv.imwrite('img_diff_threshold.png', imgThreshold)

        return imgThreshold

    def setup_params_img(parking_img: cv.Mat, parking_id, weather, spaces: list[Space], initial_params: DetectionParams = None) -> DetectionParams:

        def on_trackbar(a):
            pass

        if initial_params is None:
            init_gb_k = 3
            init_median_k = 3
            init_bw_size = 20
            init_diff = 30
        else:
            init_gb_k = initial_params.gb_k[0]
            init_median_k = initial_params.median_k
            init_bw_size = initial_params.bw_size
            init_diff = initial_params.diff_threshold

        imgGray = cv.cvtColor(parking_img, cv.COLOR_BGR2GRAY)
        img_empty = OccupancyDetectorDiff.get_empty_img(parking_id, weather)
        cv.imshow(f'img empty {parking_id}-{weather}', img_empty)

        try:
            cv.namedWindow("Trackbars")
            cv.createTrackbar("1-Gauss Kernel", "Trackbars",
                              init_gb_k, 25, on_trackbar)
            cv.createTrackbar("2-DiffThrehsold", "Trackbars",
                              init_diff, 255, on_trackbar)
            cv.createTrackbar("3-Median Kernel", "Trackbars",
                              init_median_k, 25, on_trackbar)
            cv.createTrackbar("4-BW Threshold", "Trackbars",
                              init_bw_size, 500, on_trackbar)

            cv.imshow("0 - IMG", parking_img)
            cv.waitKey(1)

            while True:
                gauss_kernel_size = cv.getTrackbarPos(
                    "1-Gauss Kernel", "Trackbars")
                if gauss_kernel_size % 2 == 0:
                    gauss_kernel_size = gauss_kernel_size+1

                if gauss_kernel_size >= 3:
                    imgBlur = cv.GaussianBlur(
                        imgGray, (gauss_kernel_size, gauss_kernel_size), 0)
                    img_empty_blur = cv.GaussianBlur(
                        img_empty, (gauss_kernel_size, gauss_kernel_size), 0)

                else:
                    imgBlur = imgGray

                diff_threshold = cv.getTrackbarPos(
                    "2-DiffThrehsold", "Trackbars")

                diff = cv.absdiff(imgBlur, img_empty_blur)
                imgThreshold = cv.threshold(
                    diff, diff_threshold, 255, cv.THRESH_BINARY)[1]

                median_kernel_size = cv.getTrackbarPos(
                    "3-Median Kernel", "Trackbars")

                if median_kernel_size % 2 == 0:
                    median_kernel_size = median_kernel_size+1

                # if median_kernel_size >= 3:
                #     imgMedian = cv.medianBlur(imgThreshold, median_kernel_size)
                # else:
                #     imgMedian = imgThreshold

                # imgEro = cv.erode(imgMedian, kernel, iterations=1)
                # imgDilate = cv.dilate(imgEro, kernel, iterations=1)

                bwThresh = cv.getTrackbarPos("4-BW Threshold", "Trackbars")
                imgBw = OccupancyDetectorDiff.bwareaopen(
                    imgThreshold, bwThresh)

                cv.imshow("1 - IMG Blur", imgBlur)
                cv.imshow("2 - IMGTresh", imgThreshold)
                cv.imshow("3 - IMG BW", imgBw)
                # cv.imshow("IMG Dilate", imgDilate)

                params = DetectionParams((gauss_kernel_size, gauss_kernel_size), gb_s=0,
                                         median_k=median_kernel_size, bw_size=bwThresh, diff_threshold=diff_threshold)

                # wait for ESC key to exit and terminate program
                key = cv.waitKey(20)
                if key == 27:
                    cv.destroyAllWindows()
                    return params

                # Run detection
                elif key == 'd':
                    detector: OccupancyDetectorDiff(params)
                    params.show_imshow = True
                    detector.detect_image(parking_img, datetime.now(), spaces)
                    params.show_imshow = False

        except Exception as e:
            print(e)
            # cv.destroyAllWindows()

    @staticmethod
    def is_space_vacant(roi: cv.Mat, space: Space, vacant_threshold) -> bool:
        """ Determine if space is vacant depending on the pixel count.
    If count is less than vacant_threshold * area portion, space is vacant.
    """
        # Count pixels with value '1' in ROI
        count = cv.countNonZero(roi)

        # Get parking space area
        vertex = space.vertex.reshape(-1, 1, 2)
        cols = vertex[:, :, 0].flatten()
        rows = vertex[:, :, 1].flatten()
        points = list(zip(cols, rows))
        a = OccupancyDetectorDiff.area(points)

        # Save area and count for later use in debug imshow
        space.area = a
        space.count = count

        # Decide if vacant using threshold
        return (count < vacant_threshold * a)

    def bwareaopen(img_input: cv.Mat, min_size: int, connectivity=8):
        """
        https://stackoverflow.com/questions/2348365/matlab-bwareaopen-equivalent-function-in-opencv
        Remove small objects from binary image (approximation of
        bwareaopen in Matlab for 2D images).

        Args:
            img: a binary image (dtype=uint8) to remove small objects from
            min_size: minimum size (in pixels) for an object to remain in the image
            connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).

        Returns:
            the binary image with small objects removed
        """
        img = img_input.copy()
        # Find all connected components (called here "labels")
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            img, connectivity=connectivity)

        # check size of all connected components (area in pixels)
        for i in range(num_labels):
            label_size = stats[i, cv.CC_STAT_AREA]

            # remove connected components smaller than min_size
            if label_size < min_size:
                img[labels == i] = 0

        return img

    # https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon
    @staticmethod
    def area(p):
        return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in OccupancyDetectorDiff.segments(p)))

    @staticmethod
    def segments(p):
        return zip(p, p[1:] + [p[0]])
