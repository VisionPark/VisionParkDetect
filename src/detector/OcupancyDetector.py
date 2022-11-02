from abc import ABC, abstractmethod
from datetime import datetime
import cv2 as cv
from src.detector.entity.DetectionParams import DetectionParams
from src.data.entity.Space import Space
import numpy as np
import cvzone


class OccupancyDetector(ABC):

    def __init__(self,  params: DetectionParams):
        self.params = params

    @abstractmethod
    def detect_image(self, parking_img: cv.Mat,  parking_img_date: datetime, spaces: list[Space]):
        return

    @staticmethod
    def imshow(img, title=''):
        while(1):
            cv.imshow(title, img)
            if cv.waitKey(20) & 0xFF == 27:
                break

    @staticmethod
    def on_trackbar(a):
        pass

    @staticmethod
    def imhist(src):
        bgr_planes = cv.split(src)
        histSize = 256
        histRange = (0, 256)  # the upper boundary is exclusive
        accumulate = False
        b_hist = cv.calcHist(bgr_planes, [0], None, [
            histSize], histRange, accumulate=accumulate)
        g_hist = cv.calcHist(bgr_planes, [1], None, [
            histSize], histRange, accumulate=accumulate)
        r_hist = cv.calcHist(bgr_planes, [2], None, [
            histSize], histRange, accumulate=accumulate)
        hist_w = 512
        hist_h = 400
        bin_w = int(round(hist_w/histSize))
        histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h,
                     norm_type=cv.NORM_MINMAX)
        cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h,
                     norm_type=cv.NORM_MINMAX)
        cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h,
                     norm_type=cv.NORM_MINMAX)
        for i in range(1, histSize):
            cv.line(histImage, (bin_w*(i-1), hist_h - int(b_hist[i-1])),
                    (bin_w*(i), hist_h - int(b_hist[i])),
                    (255, 0, 0), thickness=2)
            cv.line(histImage, (bin_w*(i-1), hist_h - int(g_hist[i-1])),
                    (bin_w*(i), hist_h - int(g_hist[i])),
                    (0, 255, 0), thickness=2)
            cv.line(histImage, (bin_w*(i-1), hist_h - int(r_hist[i-1])),
                    (bin_w*(i), hist_h - int(r_hist[i])),
                    (0, 0, 255), thickness=2)
        cv.imshow('Source image', src)
        cv.imshow('calcHist Demo', histImage)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def drawSpaceSeg(img, vertex, count, occupied, col_min, row_max, space_area, real_occupied=None):
        if real_occupied is not None:
            if not occupied and not real_occupied:  # True positive
                cv.polylines(img, [vertex], True, (0, 255, 0), thickness=2)
            elif occupied and real_occupied:  # True negative
                cv.polylines(img, [vertex], True, (0, 0, 255), thickness=2)
            elif occupied and not real_occupied:  # False negative
                cv.polylines(img, [vertex], True, (58, 146, 255), thickness=2)
            else:  # False positive
                cv.polylines(img, [vertex], True, (98, 169, 36), thickness=2)
        else:
            if not occupied:
                cv.polylines(img, [vertex], True, (0, 255, 0), thickness=2)
            else:
                cv.polylines(img, [vertex], True, (0, 0, 255), thickness=2)

        # Pixel count
        text = str(round(count/space_area, 3))
        cvzone.putTextRect(img, text, (col_min, row_max-3),
                           scale=0.8, thickness=1, offset=0)
        return img

    @staticmethod
    def get_roi(img, vertex):
        # https://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-non-rectangular-region
        cols = vertex[:, :, 0].flatten()
        rows = vertex[:, :, 1].flatten()

        points = list(zip(cols, rows))
        row_min = min(rows)
        row_max = max(rows)
        col_min = min(cols)
        col_max = max(cols)

        # print(row_min, row_max, col_min, col_max)
        mask = np.zeros(img.shape, dtype=np.uint8)
        # fill the ROI so it doesn't get wiped out when the mask is applied
        cv.fillPoly(mask, [np.array(points)], 255)

        # apply the mask
        masked_image = cv.bitwise_and(img, mask)

        return masked_image[row_min:row_max, col_min:col_max, ]
