import torch
from image import format_image


class ModelLoader:

    def __init__(self, path):
        self.model = torch.load(path)

    def test_image(self, path):
        image = format_image(path)
        tensor = self.model(image)
        value, index = torch.max(tensor[0],0)
        return index.item()
