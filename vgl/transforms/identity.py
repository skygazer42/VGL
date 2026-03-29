from vgl.transforms.base import BaseTransform


class IdentityTransform(BaseTransform):
    def __call__(self, graph):
        return graph
