
from src.data.ParkingProvider import ParkingProvider
import json

import ParkingProvider
from src.data.entity.Space import Space
import cv2 as cv
import numpy as np
import sqlite3
from datetime import datetime


class ParkingProviderSqlite(ParkingProvider):

    class ParkingProviderSqliteParams(ParkingProvider.ParkingProviderParams):
        def __init__(self, parking_id, db_file):
            super().__init__(parking_id)
            self.db_file = db_file

    def __init__(self, params: ParkingProvider.ParkingProviderSqliteParams):
        db_file = params.db_file  # Path to sqlite database file
        parking_id = params.parking_id
        con = sqlite3.connect(db_file)

    def fetch_image(self) -> cv.Mat:
        # TODO
        if self.parking_id == 'UFPR04':
            return cv.imread('E:\OneDrive - UNIVERSIDAD DE HUELVA\TFG\VisionParkDetect\dataset/UFPR04_sample.jpg')
        elif self.parking_id == 'UFPR05':
            return cv.imread('E:\OneDrive - UNIVERSIDAD DE HUELVA\TFG\VisionParkDetect\dataset/UFPR05.jpg')
        elif self.parking_id == 'PUCPR':
            return cv.imread('E:\OneDrive - UNIVERSIDAD DE HUELVA\TFG\VisionParkDetect\dataset/PUCPR_sample.jpg')
        else:
            return cv.Mat()

    def fetch_spaces(self) -> list[Space]:
        spaces_list = []
        cursorObj = self.con.cursor()
        cursorObj.execute(
            f'SELECT id,shortName,vertex,vacant,since FROM manageParking_space WHERE parking_id={self.parking_id}')

        # [(5303, '[[854.5, 219.5], [809.5, 202.5], [845.5, 194.5], [890.5, 213.5]]', True),  ...]
        spaces_db = cursorObj.fetchall()

        for space in spaces_db:
            id = json.loads(space[0])
            short_name = json.loads(space[1])
            vertex = np.array(json.loads(space[2]), dtype=np.int32)
            is_vacant = json.loads(space[3])
            since = json.loads(space[4])

            spaces_list.append(
                Space(id, vertex, short_name, is_vacant, since))

        return spaces_list

    def update_spaces_occupancy(self, spaces: list[Space]):
        for space in spaces:
            cursorObj = self.con.cursor()
            sql = f'UPDATE manageParking_space SET vacant="{1 if space.is_vacant else 0}", since=\"{space.since}\" WHERE id={space.id}'
            res = cursorObj.execute(sql)
            self.con.commit()
