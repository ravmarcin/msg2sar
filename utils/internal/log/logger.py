import logging


def get_logger() -> logging.Logger:
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger: logging.Logger = logging.getLogger(__name__)
    return logger
