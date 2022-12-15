import cv2 as cv
from src.data.entity.Space import Space
from datetime import datetime
from sys import path
path.append("../../")


class Parking:

    def __init__(self, id, name: str, spaces: list[Space], image: cv.Mat, image_date: datetime):
        self.id = id
        self.name = name
        self.spaces = spaces
        self.image = image
        self.image_date = image_date

    def __str__(self):
        return str(self.id) + ' - ' + str(self.image_date)
