from datetime import datetime
from src.data.ParkingProvider import NoSpacesException, NoImageException, ParkingProviderParams
from src.data.ParkingProvider import ParkingProvider
import json
from src.data.entity.Space import Space
import cv2 as cv
import numpy as np
import sqlite3


class ParkingProviderSqliteParams(ParkingProviderParams):
    def __init__(self, parking_id, db_file):
        super().__init__(parking_id)
        self.db_file = db_file


class ParkingProviderSqlite(ParkingProvider):

    def __init__(self, params: ParkingProviderSqliteParams):
        super().__init__(params)
        self.db_file = params.db_file  # Path to sqlite database file
        self.parking_id = params.parking_id
        self.con = sqlite3.connect(self.db_file)

    def fetch_image(self):
        # TODO
        if self.parking_id == 'UFPR04':
            return [cv.imread('E:\OneDrive - UNIVERSIDAD DE HUELVA\TFG\VisionParkDetect\dataset\\UFPR04_sample.jpg'), datetime.now()]
        elif self.parking_id == 'UFPR05':
            return [cv.imread('E:\OneDrive - UNIVERSIDAD DE HUELVA\TFG\VisionParkDetect\dataset\\UFPR05_cloudy.jpg'), datetime.now()]
        elif self.parking_id == 'PUCPR':
            return [cv.imread('E:\OneDrive - UNIVERSIDAD DE HUELVA\TFG\VisionParkDetect\dataset\\PUCPR_rainy.jpg'), datetime.now()]
        else:
            raise NoImageException('parking_id not valid')

    def fetch_spaces(self) -> list[Space]:
        spaces_list = []
        cursorObj = self.con.cursor()
        request = f'SELECT id,shortName,vertex,vacant,since FROM manageParking_space WHERE parking_id=(SELECT id FROM manageParking_parking WHERE name="{self.parking_id}")'
        cursorObj.execute(request)

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

    def get_parking_name(self, parking_id) -> str:
        return str(parking_id)
