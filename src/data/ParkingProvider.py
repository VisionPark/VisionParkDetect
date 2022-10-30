# https://stackoverflow.com/questions/2124190/how-do-i-implement-interfaces-in-python
from abc import ABC, abstractmethod
from src.data.entity.Parking import Parking
from src.data.entity.Space import Space
import cv2 as cv
from datetime import datetime


class ParkingProvider(ABC):

    class ParkingProviderParams(ABC):
        def __init__(self, parking_id):
            self.parking_id = parking_id

    @abstractmethod
    def __init__(self, params: ParkingProviderParams):
        pass

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
