

from src.detector.entity.DetectionParams import DetectionParams
from src.data.entity.Space import Space
from src.detector.OcupancyDetector import OccupancyDetector
import cv2 as cv
from datetime import datetime


class OccupancyDetectorBorders(OccupancyDetector):

    def detect_image(self, parking_img: cv.Mat, parking_img_date: datetime, spaces: list[Space]):

        imgPre = self.preProcess(parking_img, self.params)

        for space in spaces:
            vertex = space.vertex
            if(vertex.size == 0):
                continue
            vertex = vertex.reshape(-1, 1, 2)

            cols = vertex[:, :, 0].flatten()
            rows = vertex[:, :, 1].flatten()
            points = list(zip(cols, rows))
            roi = super().get_roi(imgPre, vertex)

            # Count pixels with value '1'
            count = cv.countNonZero(roi)

            # Decide if vacant depending on the detection area
            is_vacant = self.is_space_vacant(
                points, count, self.params.vacant_threshold)

            # Update space occupancy
            space.is_vacant = is_vacant
            space.since = parking_img_date

    def is_space_vacant(self, vertex, count, vacant_threshold) -> bool:
        """ Determine if space is vacant depending on the pixel count.
    If count is less than vacant_threshold * area portion, space is vacant.
    """
        a = self.area(vertex)

        return(count < vacant_threshold * a)

    def preProcess(self, img: cv.Mat) -> cv.Mat:
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

        if(self.params.channel == "l"):
            imgGray = l
        elif(self.params.channel == "v"):
            imgGray = v

        if(self.params.gb_k == None):
            imgBlur = imgGray
        else:
            imgBlur = cv.GaussianBlur(
                imgGray, self.params.gb_k, self.params.gb_s)

        imgThreshold = cv.adaptiveThreshold(
            imgBlur, 255, self.params.at_method, cv.THRESH_BINARY_INV, self.params.at_blockSize, self.params.at_C)
        # a,imgThreshold2 = cv.threshold(imgBlur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        # cv.imshow("IMGTresh", imgThreshold)

        # Remove salt and pepper noise
        if(self.params.median_k != -1):
            imgMedian = cv.medianBlur(imgThreshold, self.params.median_k)
            if(self.params.show_imshow):
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
        if(self.params.bw_size != -1):
            imgBw = self.bwareaopen(imgMedian, self.params.bw_size)
            if(self.params.show_imshow):
                cv.imshow("4 - imgBw", imgBw)
        else:
            imgBw = imgMedian
        # cv.imshow("IMG Dilate", imgDilate)

        return imgBw

    def bwareaopen(img_input, min_size, connectivity=8):
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
    def area(self, p):
        return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in self.segments(p)))

    def segments(self, p):
        return zip(p, p[1:] + [p[0]])
