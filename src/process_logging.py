"""
Logging with multiprocessing

Adapted from https://fanchenbao.medium.com/python3-logging-with-multiprocessing-f51f460b8778
"""

import logging
from logging import handlers

def listener_configurer():
    root = logging.getLogger()
    file_handler = handlers.RotatingFileHandler('mptest.log', 'a', 300, 10)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.setLevel(logging.DEBUG)

def listener_process(queue):
    listener_configurer()
    while True:  # TODO replace this with other condition
        while not queue.empty():
            record = queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)
        sleep(1)

