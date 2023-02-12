from sys import path
path.append("../../")


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
        - show_imshow :         Show debug windows)
        - diff_threshold :      Intensity difference treshold for creating binary image
    """

    def __init__(self, gb_k=(3, 3), gb_s=0, at_method=None, at_blockSize=None, at_C=None, median_k=3, bw_size=50, bw_conn=8, channel="v", vacant_threshold=0.3, show_imshow=False, diff_threshold=50, match_histograms=True):
        self.gb_k = gb_k  # GaussianBlur kernel
        self.gb_s = gb_s  # GaussianBlur sigma (std. deviation)
        self.at_method = at_method  # adaptiveThreshold method
        # adaptiveThreshold blockSize neighborhood that is used to calculate a threshold value for the pixel
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
        # Intensity difference treshold for creating binary image
        self.diff_threshold = diff_threshold

        self.match_histograms = match_histograms

    def __str__(self):
        str1 = f'DetectionParams\ngb_k: {self.gb_k}\ngb_s: {self.gb_s}\nat_method: {self.at_method}\nat_blockSize: {self.at_blockSize}\nat_C: {self.at_C}\nmedian_k: {self.median_k}\nbw_size: {self.bw_size}\nbw_conn: {self.bw_conn}\nchannel: {self.channel}\nvacant_threshold: {self.vacant_threshold}\nshow_imshow: {self.show_imshow}\ndiff_threshold={self.diff_threshold}'
        str2 = f'DetectionParams(gb_k={self.gb_k}, gb_s={self.gb_s}, at_method=cv.ADAPTIVE_THRESH_GAUSSIAN_C, at_blockSize={self.at_blockSize}, at_C={self.at_C}, median_k={self.median_k}, bw_size={self.bw_size}, vacant_threshold={self.vacant_threshold}, show_imshow={self.show_imshow}, diff_threshold={self.diff_threshold})'
        return str(str1 + '\n\n' + str2)
