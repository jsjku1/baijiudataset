
import logging
from logging.handlers import RotatingFileHandler


def get_rotate_logger(name: str, level: int = logging.INFO):
    """
    :param name:
    :param level:
    :return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = RotatingFileHandler(
        filename=f'{name}.log',
        maxBytes=1024 * 1024 * 10,
        backupCount=5
    )
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(handler)
    return logger
