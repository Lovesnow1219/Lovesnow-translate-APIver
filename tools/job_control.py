# -*- coding: utf-8 -*-
import threading

_LOCK = threading.Lock()
_CURRENT = None
_LIVE = set()
_PENDING_STOP = False
_TLS = threading.local()


class JobStopped(BaseException):
    """Cancel a run. Subclass of BaseException so `except Exception` cannot swallow it."""


def start_job():
    global _CURRENT, _PENDING_STOP
    event = threading.Event()
    with _LOCK:
        previous = _CURRENT
        _CURRENT = event
        _LIVE.add(event)
        _PENDING_STOP = False
    if previous is not None:
        previous.set()
    bind_job(event)
    return event


def bind_job(event):
    _TLS.event = event


def current_job():
    event = getattr(_TLS, 'event', None)
    if event is not None:
        return event
    with _LOCK:
        return _CURRENT


def finish_job(event=None):
    global _CURRENT, _PENDING_STOP
    event = event or getattr(_TLS, 'event', None)
    with _LOCK:
        if event in _LIVE:
            _LIVE.discard(event)
        if _CURRENT is event:
            _CURRENT = None
        if not _LIVE:
            _PENDING_STOP = False


def request_stop():
    global _PENDING_STOP
    with _LOCK:
        _PENDING_STOP = True
        if _CURRENT is not None:
            _CURRENT.set()
        for event in list(_LIVE):
            event.set()


def is_stopped():
    bound = getattr(_TLS, 'event', None)
    if bound is not None:
        return bound.is_set()
    with _LOCK:
        current = _CURRENT
        pending = _PENDING_STOP
    if current is not None and current.is_set():
        return True
    return bool(pending)


def check_stop(message='已中止'):
    if is_stopped():
        raise JobStopped(message)
