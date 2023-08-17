import copy


class LazyFunction:
    def __init__(self, function, **kwargs):
        self.function = function
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        selfkwargs = copy.deepcopy(self.kwargs)
        to_delete = []
        for key, value in kwargs.items():
            if key in selfkwargs:
                selfkwargs[key] = value
                to_delete += [key]
        for key in to_delete:
            del kwargs[key]

        return self.function(**selfkwargs, **kwargs)
