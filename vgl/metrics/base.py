class Metric:
    name = "metric"

    def reset(self):
        raise NotImplementedError

    def update(self, predictions, targets, **kwargs):
        del kwargs
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError
