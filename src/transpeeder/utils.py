import io
import os
import json

import torch.distributed as dist
from loguru import logger as logger


logger.add(f'ds_training.log')


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


class LoggerRank0:
    def trace(self, *args, **kwargs):
        if not is_rank_0():
            return
        logger.trace(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if not is_rank_0():
            return
        logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        if not is_rank_0():
            return
        logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        if not is_rank_0():
            return
        logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        if not is_rank_0():
            return
        logger.error(*args, **kwargs)

logger_rank0 = LoggerRank0()


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
