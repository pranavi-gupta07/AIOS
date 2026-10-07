__all__ = ["AIOSError", "Client", "Result", "infer"]
__version__ = "0.1.0"


def __getattr__(name):
    if name in __all__:
        from . import api
        return getattr(api, name)
    raise AttributeError(name)
