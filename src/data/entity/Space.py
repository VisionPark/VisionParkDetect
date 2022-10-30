from datetime import datetime
import numpy as np


class Space:
    """_summary_Space
    """

    def __init__(self, id, vertex=None, short_name: str = id, is_vacant: bool = False, since: datetime = datetime.now(), is_vacant_real=None):
        id = id
        short_name = short_name
        is_vacant_real = is_vacant_real
        is_vacant = is_vacant
        since = since
        if vertex == None:
            vertex = np.array([])
        else:
            vertex = vertex
