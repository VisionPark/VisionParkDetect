
from math import floor
from src.data.ParkingProvider import NoSpacesException, NoImageException
from datetime import datetime
from src.data.ParkingProvider import ParkingProvider, ParkingProviderParams
from src.data.ParkingProviderLocal import ParkingProviderLocal, ParkingProviderLocalParams
from src.data.entity.Parking import Parking
from src.data.entity.Space import Space
import cv2 as cv
import glob
import random
from bs4 import BeautifulSoup  # read XML
import numpy as np
import sqlite3
import json

from sys import path
path.append("../")


class ParkingProviderLocalSqliteParams(ParkingProviderLocalParams):
    def __init__(self, parking_id, path, k, random_seed=None, db_file=None):
        super().__init__(parking_id, path, k, random_seed)
        self.db_file = db_file


class ParkingProviderLocalSqlite(ParkingProviderLocal):

    def __init__(self, params: ParkingProviderLocalSqliteParams):
        super().__init__(params)

        self.con = sqlite3.connect(params.db_file, timeout=10)
        self.parking_id = params.parking_id

    # def fetch_image(self) -> tuple[cv.Mat, datetime]:
    #     # Read and return next image
    #     if self.index < len(self.img_files):
    #         img = cv.imread(self.img_files[self.index])
    #         self.index += 1
    #         return img, datetime.now()
    #     else:
    #         index = 0
    #         raise NoImageException('Finished fetching path')

    def fetch_spaces(self) -> list[Space]:
        spaces_list = []
        cursorObj = self.con.cursor()
        request = f'SELECT id,shortName,vertex,vacant,since FROM manageParking_space WHERE parking_id=(SELECT id FROM manageParking_parking WHERE name="{self.parking_id}")'
        cursorObj.execute(request)

        # [(5303, '[[854.5, 219.5], [809.5, 202.5], [845.5, 194.5], [890.5, 213.5]]', True),  ...]
        # (5303, 'A0', '[[854.5, 219.5], [809.5, 202.5], [845.5, 194.5], [890.5, 213.5]]', 0, '2022-09-18 12:05:24.632553')

        spaces_db = cursorObj.fetchall()

        for space in spaces_db:
            id = str(space[0])
            short_name = space[1]
            vertex = np.array(json.loads(space[2]), dtype=np.int32)
            is_vacant = bool(space[3])
            since = datetime.strptime(space[4], '%Y-%m-%d %H:%M:%S.%f')
            spaces_list.append(
                Space(id, vertex, short_name, is_vacant, since))

        return spaces_list

    def update_spaces_occupancy(self, spaces: list[Space]):
        for space in spaces:
            cursorObj = self.con.cursor()
            sql = f'UPDATE manageParking_space SET vacant="{1 if space.is_vacant else 0}", since=\"{space.since}\" WHERE id={space.id}'
            res = cursorObj.execute(sql)
            self.con.commit()

    # def get_parking_name(self, parking_id):
    #     return str(parking_id)

    # def get_parking(self) -> Parking:
    #     parking = super().get_parking()
    #     self.index += 1  # Increment index for processing next img,xml
    #     return parking

    @staticmethod
    def get_points_xml(space_xml):
        vertex = []
        for p in space_xml.contour.find_all('point'):
            vertex.append([p.get('x'), p.get('y')])
        return np.array(vertex, dtype=np.int32)

    @staticmethod
    def get_random_files(path, k) -> list[str]:
        files = glob.glob(path + '/**/*.jpg', recursive=True)
        return random.choices(files, k=k)
