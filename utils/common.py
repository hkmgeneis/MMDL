import logging
import os


def get_logger(save_path, logger_name):
    """
    Initialize logger
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    # file log
    # file_handler = RotatingFileHandler(os.path.join(save_path, "future_tracker.log"), maxBytes=2*1024*1024, backupCount=1)
    # file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = get_logger(os.getcwd(), __name__)