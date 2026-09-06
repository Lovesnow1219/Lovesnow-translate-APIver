# -*- coding: utf-8 -*-
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def env_workers(name, default=4, maximum=8):
    try:
        value = int(os.getenv(name) or default)
    except (TypeError, ValueError):
        value = default
    return max(1, min(int(maximum), value))


def map_parallel(fn, items, workers=4):
    from tools.job_control import bind_job, check_stop, current_job, is_stopped

    items = list(items)
    if not items:
        return []
    workers = max(1, min(int(workers or 1), len(items)))
    if workers <= 1:
        results = []
        for item in items:
            check_stop()
            results.append(fn(item))
        return results

    parent = current_job()

    def _run(item):
        if parent is not None:
            bind_job(parent)
        check_stop()
        return fn(item)

    results = [None] * len(items)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(_run, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            if is_stopped():
                for pending in futures:
                    pending.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                check_stop()
            results[futures[future]] = future.result()
        return results
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
