import math


class DistanceHelper:

    @staticmethod
    def euclidean(x1, y1, x2, y2):
        length = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

        return length
