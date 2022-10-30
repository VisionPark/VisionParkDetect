import datetime
import data.entity.Space as Space
import cv2 as cv


class Parking:

    def __init__(self, id, name: str, spaces: Space(), image: cv.Mat, image_date: datetime):
        self.id = id
        self.name = name
        self.spaces = spaces
        self.image = image
        self.image_date = image_date
