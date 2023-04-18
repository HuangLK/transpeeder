from loguru import logger
from common.utils import is_rank_0
from common.mp_wraps import rank_zero

# Log config
LOG_FILENAME = "ds_training.log"

class GetLogger:
    __instance = None
    __init_flag = True

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(GetLogger, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        if self.__init_flag:
            logger.add(LOG_FILENAME)
            self.__init_flag: False

    @rank_zero
    def trace(self, *args, **kwargs):
        logger.trace(*args, **kwargs)

    @rank_zero
    def debug(self, *args, **kwargs):
        logger.debug(*args, **kwargs)

    @rank_zero
    def info(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    @rank_zero
    def warning(self, *args, **kwargs):
        logger.warning(*args, **kwargs)

    @rank_zero
    def error(self, *args, **kwargs):
        logger.error(*args, **kwargs)

logger_rank0 = GetLogger()