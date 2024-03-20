import logging
import colorlog

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = colorlog.ColoredFormatter(
    '%(asctime)s%(log_color)s [%(levelname).1s]%(reset)s [%(filename)s-%(lineno)d] %(message)s',
    datefmt='%m%d/%H:%M:%S',
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow,bold',
        'ERROR': 'red,bold',
        'CRITICAL': 'red,bg_yellow,bold'
    },
    secondary_log_colors={},
    style='%')

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
