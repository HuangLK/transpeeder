from functools import wraps
from common.utils import is_rank_0

__all__ = ["rank_zero"]


def rank_zero(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_rank_0():
            return
        result = func(*args, **kwargs)
        return result

    return wrapper