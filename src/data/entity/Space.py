from datetime import datetime
import numpy as np


class Space:
    """_summary_Space
    """

    def __init__(self, id, vertex=None, short_name: str = id, is_vacant: bool = False, since: datetime = datetime.now(), is_vacant_real=None):
        self.id = id
        self.short_name = short_name
        self.is_vacant_real = is_vacant_real
        self.is_vacant = is_vacant
        self.since = since

        if vertex is None:
            self.vertex = np.array([])
        else:
            self.vertex = vertex
