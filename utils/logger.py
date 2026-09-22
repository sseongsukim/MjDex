import logging
import colorlog

formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "white",
        "INFO": "cyan",
        "WARNING": "yellow",
        "ERROR": "red,bold",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)

Logger = colorlog.getLogger("robot-learning")
Logger.setLevel(logging.DEBUG)
Logger.propagate = False

if not Logger.handlers:
    ch = colorlog.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    Logger.addHandler(ch)
