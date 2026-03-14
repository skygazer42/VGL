class Metric:
    def __call__(self, predictions, targets):
        raise NotImplementedError
