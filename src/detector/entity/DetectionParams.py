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
