from vgl.transforms.base import BaseTransform


class Compose(BaseTransform):
    def __init__(self, transforms):
        self.transforms = tuple(transforms)

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data
