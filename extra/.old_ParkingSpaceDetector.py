import multiprocessing
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn import metrics
from bs4 import BeautifulSoup  # read XML
import cv2 as cv
import cvzone
import sqlite3
import numpy as np
import json
import os  # list directories
from datetime import datetime
from collections import namedtuple

# GETTING INFO FROM DATABASE
# DB CONNECTION
con = sqlite3.connect(
    'E:/OneDrive - UNIVERSIDAD DE HUELVA\TFG\VisionParkWeb-main\VisionParkWeb\VisionParkWeb\db.sqlite3', timeout=10)


def fetch_parkings(con) -> list:
    cursorObj = con.cursor()
    cursorObj.execute('SELECT id,name FROM manageParking_parking')

    return cursorObj.fetchall()


def get_spaces(parking_id) -> list:
    cursorObj = con.cursor()
    cursorObj.execute(
        f'SELECT id,vertex FROM manageParking_space WHERE parking_id={parking_id}')

    # [(5303, '[[854.5, 219.5], [809.5, 202.5], [845.5, 194.5], [890.5, 213.5]]'), ...]
    return cursorObj.fetchall()


def get_space_vacant(space_id) -> bool:
    cursorObj = con.cursor()
    cursorObj.execute(
        f'SELECT vacant FROM manageParking_space WHERE id={space_id}')

    return cursorObj.fetchall()[0][0]


def update_space_occupancy(space_id, is_vacant):
    now = datetime.now()
    cursorObj = con.cursor()
    sql = f'UPDATE manageParking_space SET vacant="{1 if is_vacant else 0}", since=\"{now}\" WHERE id={space_id}'
    res = cursorObj.execute(sql)
    con.commit()


# CAR DETECTION IN ROI
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


def imshow(img, title=''):
    while(1):
        cv.imshow(title, img)
        if cv.waitKey(20) & 0xFF == 27:
            break


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


def on_trackbar(a):
    pass


# https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon


def area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in segments(p)))


def segments(p):
    return zip(p, p[1:] + [p[0]])


def is_space_vacant(vertex, count, vacant_threshold) -> bool:
    """ Determine if space is vacant depending on the pixel count.
 If count is less than vacant_threshold * area portion, space is vacant.
"""
    a = area(vertex)

    return(count < vacant_threshold * a)


class DetectionParams:
    """ # Parameters used for detecting space occupancy.
        # Attributes:
        - gb_k :                GaussianBlur kernel
        - gb_s :                GaussianBlur sigma (std. deviation)
        - at_method :           adaptiveThreshold method
        - at_blockSize :        adaptiveThreshold blockSize neighborhood that is used to calculate a threshold value for the pixel
        - at_C :                adaptiveThreshold C constant to be substracted
        - median_k :            Median filter kernel size (-1 if not desired to apply)
        - bw_size :             bwareaopen remove objects smaller than this size (-1 if not desired to apply)
        - bw_conn :             bwareaopen neighborhood connectivity (default 8)
        - channel :             Color channel to use {'g'ray, hs'v' or h'l's }
        - vacant_threshold :    Threshold (0 to 1) to determine space is vacant depending on pixel count
        - show_imshow :         Show debug windows
    """

    def __init__(self, gb_k, gb_s, at_method, at_blockSize, at_C, median_k=-1, bw_size=-1, bw_conn=8, channel="v", vacant_threshold=0.3, show_imshow=False):
        self.gb_k = gb_k  # GaussianBlur kernel
        self.gb_s = gb_s  # GaussianBlur sigma (std. deviation)
        self.at_method = at_method  # adaptiveThreshold method
        # adaptiveThreshold blockSizeneighborhood that is used to calculate a threshold value for the pixel
        self.at_blockSize = at_blockSize
        self.at_C = at_C  # adaptiveThreshold C constant to be substracted
        # Median filter kernel size (-1 if not desired to apply)
        self.median_k = median_k
        # bwareaopen remove objects smaller than this size (-1 if not desired to apply)
        self.bw_size = bw_size
        # bwareaopen neighborhood connectivity (default 8)
        self.bw_conn = bw_conn
        # Color channel to use {'g'ray, hs'v' or h'l's }
        self.channel = channel
        # Threshold (0 to 1) to determine space is vacant depending on pixel count
        self.vacant_threshold = vacant_threshold
        # Show debug windows
        self.show_imshow = show_imshow


class DetectionResult:

    def __init__(self, confusion_matrix: metrics.confusion_matrix, real, predicted):
        self.confusion_matrix = confusion_matrix
        self.real = real
        self.predicted = predicted
        self.stats = None

    def __init__(self, confusion_matrix: metrics.confusion_matrix, real, predicted, stats: namedtuple):
        self.confusion_matrix = confusion_matrix
        self.real = real
        self.predicted = predicted

    def get_stats(self) -> namedtuple:
        tn, fp, fn, tp = metrics.confusion_matrix(
            self.real, self.predicted).ravel()

        # Precision Score = TP / (FP + TP). Minimize FP
        precision = tp / (fp+tp)

        # Specificity score = TN / (TN+FP)
        specificity = tn / (tn+fp)

        # Recall Score = TP / (FN + TP). Minimize FN
        recall = tp / (fn+tp)

        # F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/) . Minimize FN over minimizing FP
        f1 = 2*precision*recall / (precision + recall)

        # Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
        accuracy = (tp+tn) / (tp+fn+tn+fp)

        self.stats = namedtuple(
            precision=precision, specificity=specificity, recall=recall, f1=f1, accuracy=accuracy)
        return self.stats

    def show_confusion_matrix(self):
        if self.stats is None:
            self.get_stats()

        print('Precision: %.3f' % self.stats.precision)
        print('specificity: %.3f' % self.stats.specificity)
        print('Recall: %.3f' % self.stats.recall)
        print('F1 Score: %.3f' % self.stats.f1)
        print('Accuracy: %.3f' % self.stats.accuracy)

        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=self.confusion_matrix, display_labels=['Occupied', 'Vacant'])
        cm_display.plot()
        plt.show()


def preProcess(img: cv.Mat, params: DetectionParams) -> cv.Mat:
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
        imgBlur = cv.GaussianBlur(imgGray, params.gb_k, params.gb_s)

    imgThreshold = cv.adaptiveThreshold(
        imgBlur, 255, params.at_method, cv.THRESH_BINARY_INV, params.at_blockSize, params.at_C)
    # a,imgThreshold2 = cv.threshold(imgBlur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # cv.imshow("IMGTresh", imgThreshold)

    # Remove salt and pepper noise
    if(params.median_k != -1):
        imgMedian = cv.medianBlur(imgThreshold, params.median_k)
        if(params.show_imshow):
            cv.imshow("1 - IMGBlur", imgBlur)
            cv.imshow("2 - IMGTresh", imgThreshold)
            cv.imshow("3 - IMGMedian", imgMedian)
    else:
        imgMedian = imgThreshold

    # Make thicker edges
    # kernel = np.ones((5,5), np.uint8)
    # imgEro = cv.erode(imgMedian, kernel, iterations=1)
    # imgDilate = cv.dilate(imgEro, kernel, iterations=1)

    # Remove small objects
    if(params.bw_size != -1):
        imgBw = bwareaopen(imgMedian, params.bw_size)
        if(params.show_imshow):
            cv.imshow("4 - imgBw", imgBw)
    else:
        imgBw = imgMedian
    # cv.imshow("IMG Dilate", imgDilate)

    return imgBw


def setup_preprocess(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.namedWindow("Trackbars")
    cv.createTrackbar("1-Gauss Kernel", "Trackbars", 3, 25, on_trackbar)
    cv.createTrackbar("2-Blocksize", "Trackbars", 25, 100, on_trackbar)
    cv.createTrackbar("2-C", "Trackbars", 16, 100, on_trackbar)
    cv.createTrackbar("3-Median Kernel", "Trackbars", 3, 25, on_trackbar)
    cv.createTrackbar("4-BW Threshold", "Trackbars", 20, 500, on_trackbar)

    cv.imshow("0 - IMG", img)

    while True:
        try:
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
            imgBw = bwareaopen(imgMedian, bwThresh)

            cv.imshow("1 - IMG Blur", imgBlur)
            cv.imshow("2 - IMGTresh", imgThreshold)
            cv.imshow("3 - IMGMedian", imgMedian)
            cv.imshow("4 - IMG BW", imgBw)
            # cv.imshow("IMG Dilate", imgDilate)

            if cv.waitKey(1) == 27:         # wait for ESC key to exit and terminate progra,
                cv.destroyAllWindows()
                break
        except:
            cv.destroyAllWindows()

# SPACE DETECTION IN XML


def get_points_xml(space):
    vertex = []
    for p in space.contour.find_all('point'):
        vertex.append([p.get('x'), p.get('y')])
    return np.array(vertex, dtype=np.int32)


def detect_batch(files, params: DetectionParams, showConfusionMatrix=True):
    predicted = []
    real = []

    n_files = len(files)
    for idx, filename in enumerate(files):
        clear_output(wait=True)
        print(f"{idx+1}/{n_files}: ", filename)
        # SPACE OCCUPATION DETECTION
        parking_img = cv.imread(filename)
        img = parking_img

        imgPre = preProcess(img, params)

        # Get spaces from xml
        with open(filename.replace('.jpg', '.xml'), 'r') as f:
            file = f.read()
        data = BeautifulSoup(file, "xml")
        spaces = data.find_all('space')

        for space in spaces:
            vertex = get_points_xml(space)
            if(vertex.size == 0):
                continue
            vertex = vertex.reshape(-1, 1, 2)

            cols = vertex[:, :, 0].flatten()
            rows = vertex[:, :, 1].flatten()
            points = list(zip(cols, rows))
            roi = get_roi(imgPre, vertex)

            # cv.imshow("roi", roi)

            # Count pixels with value '1'
            count = cv.countNonZero(roi)

            # drawSpaceSeg(img, vertex, count)
            # Depending on the detection area
            vacant = is_space_vacant(points, count, params.vacant_threshold)

            vacant_real = space.get('occupied') == "0"
            predicted.append(vacant)
            real.append(vacant_real)

            if(params.show_imshow):
                space_area = area(points)
                assert(space_area > 0)
                drawSpaceSeg(img, np.array(points, np.int32), count, not vacant, min(
                    cols), max(rows), space_area, not vacant_real)
                if(vacant != vacant_real):  # Show error in prediction
                    print("ERROR PREDICTED vacant: "+str(bool(vacant)))
                    print("Pixel count: "+str(count))
                    print(f"Area: {space_area} k={count/space_area}")
                    print("---------------------------------")
                cv.namedWindow("roi")
                cv.destroyWindow("roi")
                cv.imshow("roi", roi)
                cv.imshow("IMG with space seg", img)

                key = cv.waitKey(0)
                if(key == 27):
                    break

    confusion_matrix = metrics.confusion_matrix(real, predicted)
    if(showConfusionMatrix):
        # Precision Score = TP / (FP + TP). Minimize FP
        print('Precision: %.3f' % metrics.precision_score(real, predicted))
        # Recall Score = TP / (FN + TP). Minimize FN
        print('Recall: %.3f' % metrics.recall_score(real, predicted))
        # F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/) . Minimize FN over minimizing FP
        print('F1 Score: %.3f' % metrics.f1_score(real, predicted))
        # Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
        print('Accuracy: %.3f' % metrics.accuracy_score(real, predicted))

        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=['Occupied', 'Vacant'])
        cm_display.plot()
        plt.show()

    if(params.show_imshow):
        cv.destroyAllWindows()

    return confusion_matrix


def detect_image(filename, params: DetectionParams):
    predicted = []
    real = []

    # SPACE OCCUPATION DETECTION
    parking_img = cv.imread(filename)
    img = parking_img
    imgPre = preProcess(img, params)

    # Get spaces from xml
    with open(filename.replace('.jpg', '.xml'), 'r') as f:
        file = f.read()
    data = BeautifulSoup(file, "xml")
    spaces = data.find_all('space')

    for space in spaces:
        vertex = get_points_xml(space)
        if(vertex.size == 0):
            continue
        vertex = vertex.reshape(-1, 1, 2)

        cols = vertex[:, :, 0].flatten()
        rows = vertex[:, :, 1].flatten()
        points = list(zip(cols, rows))
        roi = get_roi(imgPre, vertex)

        # Count pixels with value '1'
        count = cv.countNonZero(roi)

        # Depending on the detection area
        vacant = is_space_vacant(points, count, params.vacant_threshold)

        vacant_real = space.get('occupied') == "0"
        predicted.append(vacant)
        real.append(vacant_real)

    if(params.show_imshow):
        cv.waitKey(0)

    confusion_matrix = metrics.confusion_matrix(real, predicted)
    return confusion_matrix, real, predicted


def detect_wrapper(args):
    return detect_image(*args)


def process_batch(files, params: DetectionParams):
    num_cores = multiprocessing.cpu_count()
    print(f"Processing {len(files)} files with {num_cores} cores")

    args = [(img, params)
            for img in files]  # lista de tuplas con argumentos

    with ThreadPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(total=len(files)) as progress:
            futures = []
            for arg in args:
                future = pool.submit(detect_wrapper, arg)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            # guardamos los resultados
            confusion_matrices = list()
            real = list()
            predicted = list()

            for future in futures:
                confusion_matrix_part, real_part, predicted_part = future.result()
                confusion_matrices.append(confusion_matrix_part)
                real += real_part
                predicted += predicted_part

            # Total confusion_matrix
            def check_matrix(m): return np.shape(m) == (2, 2)
            confusion_matrices = np.array(
                list(filter(check_matrix, confusion_matrices)))
            confusion_matrix = np.sum(confusion_matrices, axis=0)
            return confusion_matrix, real, predicted
