import logging
import sys
import os

from core.utils.distributed import get_rank_and_world_size


rank, world_size = get_rank_and_world_size()
log_level = os.environ.get("MAESTRO_LOG_LEVEL", "INFO").upper()
logging.basicConfig(format=f'%(asctime)s - [{rank}] %(name)s - %(levelname)s - %(message)s', stream=sys.stdout, level=log_level)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    return logger

def get_root_rank_logger() -> logging.Logger:
    """Get a logger that is only enabled on rank 0"""
    logger = logging.getLogger()
    if rank != 0:
        logger.disabled = True
    return logger