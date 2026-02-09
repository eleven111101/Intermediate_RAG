import time
from contextlib import contextmanager

@contextmanager
def timed_block(name: str, logger):
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} took {elapsed:.2f}s")
