import time
from contextlib import contextmanager


@contextmanager
def timed_block(name: str, logger):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{name} took {(end - start):.2f}s")
