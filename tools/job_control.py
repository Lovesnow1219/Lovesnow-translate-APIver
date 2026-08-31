# -*- coding: utf-8 -*-
import threading

_LOCK = threading.Lock()
_CURRENT = None


class JobStopped(Exception):
    pass


def start_job():
    event = threading.Event()
    with _LOCK:
        global _CURRENT
        previous = _CURRENT
        _CURRENT = event
    if previous is not None:
        previous.set()
    return event


def request_stop():
    with _LOCK:
        if _CURRENT is not None:
            _CURRENT.set()


def is_stopped():
    with _LOCK:
        event = _CURRENT
    return bool(event is not None and event.is_set())


def check_stop(message='已中止'):
    if is_stopped():
        raise JobStopped(message)
