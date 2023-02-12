

from src.detector.entity.DetectionParams import DetectionParams
from src.data.entity.Space import Space
from src.detector.OcupancyDetector import OccupancyDetector
import cv2 as cv
from datetime import datetime
import numpy as np
import threading
import math


class OccupancyDetectorBorders(OccupancyDetector):

    def __init__(self, params: DetectionParams):
        super().__init__(params)

    @staticmethod
    def detect_image(params: DetectionParams, parking_img: cv.Mat, parking_img_date: datetime, spaces: list[Space]):

        imgPre = OccupancyDetectorBorders.preProcess(
            params, parking_img)
        # imgPre = OccupancyDetectorBorders.canny_edge_detection(
        #     parking_img, params, params.show_imshow)

        new_spaces = []
        for space in spaces.copy():
            vertex = space.vertex
            if(vertex.size == 0):
                continue
            vertex = vertex.reshape(4, 1, 2)

            # Get ROI of parking space
            roi = OccupancyDetector.get_roi(imgPre, vertex)

            # Decide if vacant depending on the detection area
            is_vacant = OccupancyDetectorBorders.is_space_vacant(
                roi, space, params.vacant_threshold)

            # Update space occupancy
            space.is_vacant = is_vacant
            space.since = parking_img_date
            new_spaces.append(space)

        return new_spaces

    @staticmethod
    def canny_edge_detection(img, params, show_imshow=False):
        # Convert the image to grayscale
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if(params.gb_k is None):
            imgBlur = imgGray
        else:
            imgBlur = cv.GaussianBlur(
                imgGray, params.gb_k, params.gb_s)

        if(params.bf_d is None):
            imgBlur = imgGray
        else:
            imgBlur = cv.bilateralFilter(
                imgBlur, params.bf_d, params.bf_sigma_color, params.bf_sigma_space)

        # Apply Canny edge detection
        if(params.dynamic_t is None):
            t1 = params.t1
            t2 = params.t2
        else:
            v = np.median(imgGray)
            sigma = 0.33

            # ---- apply optimal Canny edge detection using the computed median----
            t1 = int(max(0, (1.0 - sigma) * v))
            t2 = int(min(255, (1.0 + sigma) * v))
            print(t1, t2)

        imgEdges = cv.Canny(imgBlur, params.t1, params.t2)

        # Remove small objects
        if(params.bw_size != -1):
            imgBw = OccupancyDetectorBorders.bwareaopen(
                imgEdges, params.bw_size)
        else:
            imgBw = imgEdges
        # cv.imshow("IMG Dilate", imgDilate)

        if(show_imshow):
            cv.imshow('1 - Img gray', imgGray)
            cv.imshow('2 - Biltareal filter', imgBlur)
            cv.imshow('3 - Canny Edge Detection', imgEdges)
            cv.imshow("4 - imgBw", imgBw)

            # # Use Hough transform to detect lines in the edge map
            # lines = cv.HoughLinesP(
            #     imgEdges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

            # # Iterate over the detected lines and draw them on the image
            # if lines is not None:
            #     for line in lines:
            #         x1, y1, x2, y2 = line[0]
            #         cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # # Show the resulting image
            # cv.imshow("Image with detected lines", img)

        return imgBw

    def setup_params_img(parking_img: cv.Mat, spaces: list[Space], initial_params: DetectionParams = None) -> DetectionParams:

        def on_trackbar(a):
            pass

        if initial_params is None:
            init_gb_k = 3
            init_at_blockSize = 25
            init_at_c = 16
            init_median_k = 3
            init_bw_size = 20
        else:
            init_gb_k = initial_params.gb_k[0]
            init_at_blockSize = initial_params.at_blockSize
            init_at_c = initial_params.at_C
            init_median_k = initial_params.median_k
            init_bw_size = initial_params.bw_size

        imgGray = cv.cvtColor(parking_img, cv.COLOR_BGR2GRAY)
        try:
            cv.namedWindow("Trackbars")
            cv.createTrackbar("1-Gauss Kernel", "Trackbars",
                              init_gb_k, 25, on_trackbar)
            cv.createTrackbar("2-Blocksize", "Trackbars",
                              init_at_blockSize, 100, on_trackbar)
            cv.createTrackbar("2-C", "Trackbars", init_at_c, 100, on_trackbar)
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
                else:
                    imgBlur = imgGray

                blocksize = cv.getTrackbarPos("2-Blocksize", "Trackbars")
                c = cv.getTrackbarPos("2-C", "Trackbars")
                if blocksize % 2 == 0 or blocksize == 0:
                    blocksize = blocksize+1

                imgThreshold = cv.adaptiveThreshold(
                    imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blocksize, c)
                # imgThreshold2 = cv.threshold(imgBlur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

                median_kernel_size = cv.getTrackbarPos(
                    "3-Median Kernel", "Trackbars")

                if median_kernel_size % 2 == 0:
                    median_kernel_size = median_kernel_size+1

                if median_kernel_size >= 3:
                    imgMedian = cv.medianBlur(imgThreshold, median_kernel_size)
                else:
                    imgMedian = imgThreshold

                # imgEro = cv.erode(imgMedian, kernel, iterations=1)
                # imgDilate = cv.dilate(imgEro, kernel, iterations=1)

                bwThresh = cv.getTrackbarPos("4-BW Threshold", "Trackbars")
                imgBw = OccupancyDetectorBorders.bwareaopen(
                    imgMedian, bwThresh)

                cv.imshow("1 - IMG Blur", imgBlur)
                cv.imshow("2 - IMGTresh", imgThreshold)
                cv.imshow("3 - IMGMedian", imgMedian)
                cv.imshow("4 - IMG BW", imgBw)
                # cv.imshow("IMG Dilate", imgDilate)

                params = DetectionParams((gauss_kernel_size, gauss_kernel_size), gb_s=0, at_method=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         at_blockSize=blocksize, at_C=c, median_k=median_kernel_size, bw_size=bwThresh)

                # wait for ESC key to exit and terminate program
                key = cv.waitKey(20)
                if key == 27:
                    cv.destroyAllWindows()
                    return params

                # Run detection
                elif key == 'd':
                    detector: OccupancyDetectorBorders(params)
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
        a = OccupancyDetectorBorders.area(points)

        # Save area and count for later use in debug imshow
        space.area = a
        space.count = count

        # Decide if vacant using threshold
        return(count < vacant_threshold * a)

    @staticmethod
    def preProcess(params: DetectionParams, img: cv.Mat) -> cv.Mat:
        # imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        imgHLS = cv.cvtColor(img, cv.COLOR_BGR2HLS)
        l = imgHLS[:, :, 1]
        imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        v = imgHSV[:, :, 2]
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # cv.imshow("l", l)
        # cv.imshow("v", v)
        # cv.imshow("imgGray", imgGray)
        # cv.waitKey(0)

        if(params.channel == "l"):
            imgGray = l
        elif(params.channel == "v"):
            imgGray = v

        if(params.gb_k is None):
            imgBlur = imgGray
        else:
            imgBlur = cv.GaussianBlur(
                imgGray, params.gb_k, params.gb_s)

        # if(params.bf_d is not None):
        #     imgBlur = cv.bilateralFilter(
        #         imgBlur, params.bf_d, params.bf_sigma_color, params.bf_sigma_space)

        imgThreshold = cv.adaptiveThreshold(
            imgBlur, 255, params.at_method, cv.THRESH_BINARY_INV, params.at_blockSize, params.at_C)
        # a,imgThreshold2 = cv.threshold(imgBlur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        # cv.imshow("IMGTresh", imgThreshold)

        # Remove salt and pepper noise
        if(params.median_k != -1):
            imgMedian = cv.medianBlur(imgThreshold, params.median_k)
        else:
            imgMedian = imgThreshold

        # Make thicker edges
        # kernel = np.ones((5,5), np.uint8)
        # imgEro = cv.erode(imgMedian, kernel, iterations=1)
        # imgDilate = cv.dilate(imgEro, kernel, iterations=1)

        # Remove small objects
        if(params.bw_size != -1):
            imgBw = OccupancyDetectorBorders.bwareaopen(
                imgMedian, params.bw_size)
        else:
            imgBw = imgMedian
        # cv.imshow("IMG Dilate", imgDilate)

        # Show images
        if(params.show_imshow):
            cv.imshow("1 - IMGBlur", imgBlur)
            cv.imshow("2 - IMGTresh", imgThreshold)
            cv.imshow("3 - IMGMedian", imgMedian)
            cv.imshow("4 - imgBw", imgBw)

        return imgBw

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
                         for ((x0, y0), (x1, y1)) in OccupancyDetectorBorders.segments(p)))

    @staticmethod
    def segments(p):
        return zip(p, p[1:] + [p[0]])
