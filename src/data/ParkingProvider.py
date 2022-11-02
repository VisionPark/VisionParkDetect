# https://stackoverflow.com/questions/2124190/how-do-i-implement-interfaces-in-python
from datetime import datetime
import cv2 as cv
from src.data.entity.Space import Space
from src.data.entity.Parking import Parking
from abc import ABC, abstractmethod
from sys import path
path.append("../")


class ParkingProviderParams(ABC):
    def __init__(self, parking_id):
        self.parking_id = parking_id


class ParkingProvider(ABC):

    def __init__(self, params: ParkingProviderParams):
        self.params = params

    @abstractmethod
    def fetch_image(self) -> tuple[cv.Mat, datetime]:
        pass

    @abstractmethod
    def fetch_spaces(self) -> list[Space]:
        pass

    @abstractmethod
    def update_spaces_occupancy(self, new_spaces: list[Space]):
        pass

    @abstractmethod
    def get_parking_name(self, parking_id) -> str:
        pass

    def get_parking(self) -> Parking:
        name = self.get_parking_name(self.params.parking_id)
        [img, img_date] = self.fetch_image()
        spaces = self.fetch_spaces()
        return Parking(self.params.parking_id, name, spaces, img, img_date)
