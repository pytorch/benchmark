

import logging 

LOGGER_NAME = 'TorchBenchLogger'


def set_logger(logger_level=logging.INFO):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logger_level)
    logger.addHandler(handler)
    return logger