
from src.data.ParkingProvider import NoSpacesException, NoImageException
from datetime import datetime
from src.data.ParkingProvider import ParkingProvider, ParkingProviderParams
from src.data.entity.Space import Space
import cv2 as cv
import glob
import random
from bs4 import BeautifulSoup  # read XML
import numpy as np
from sys import path
path.append("../")


class ParkingProviderLocalParams(ParkingProviderParams):
    def __init__(self, parking_id, path, k):
        super().__init__(parking_id)
        self.path = path
        self.k = k


class ParkingProviderLocal(ParkingProvider):

    def __init__(self, params: ParkingProviderLocalParams):
        super().__init__(params)
        path = params.path  # Path to folder with images and spaces files
        self.k = params.k   # 1/k random files to select
        self.index = 0

        files = glob.glob(path + '/**/*.jpg', recursive=True)
        self.img_files = random.choices(files, k=int(len(files)/self.k))
        self.spaces_files = [file.replace('.jpg', '.xml')
                             for file in self.img_files]
        self.num_files = len(self.img_files)
        print(f'Selected {self.num_files} files')

    def get_num_files(self):
        return self.num_files

    def fetch_image(self) -> tuple[cv.Mat, datetime]:
        # Read and return next image
        if self.index < len(self.img_files):
            img = cv.imread(self.img_files[self.index])
            self.index += 1
            return img, datetime.now()
        else:
            raise NoImageException('Finished fetching path')

    def fetch_spaces(self) -> list[Space]:

        if self.index < len(self.spaces_files):
            spaces_list = []

            # Open XML file
            with open(self.spaces_files[self.index], 'r') as f:
                file = f.read()
            data = BeautifulSoup(file, "xml")

            # Read spaces node
            spaces_node = data.find_all('space')
            for space_xml in spaces_node:
                # Extract vertex from xml
                vertex = self.get_points_xml(space_xml)

                # Get space id from xml
                id = space_xml.get('id')

                # Get real occupancy status
                is_vacant_real = space_xml.get('occupied') == "0"

                # Build the space object
                new_space = Space(id, vertex)
                new_space.is_vacant_real = is_vacant_real

                spaces_list.append(new_space)

            return spaces_list

        else:
            raise NoSpacesException('Finished fetching path')

    def update_spaces_occupancy(self, spaces: list[Space]):
        pass

    def get_parking_name(self, parking_id):
        return str(parking_id)

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
