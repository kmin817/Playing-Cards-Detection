import cv2

class NameImg:

    name: str
    img: cv2.typing.MatLike


    def __init__(self, name: str, img: cv2.typing.MatLike) -> None:
        self.name = name
        self.img = img
