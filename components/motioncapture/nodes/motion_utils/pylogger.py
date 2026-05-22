from time import time
import logging
import torch


def sync_time():
    torch.cuda.synchronize()
    return time()


Log = logging.getLogger("motioncapture")
Log.time = time
Log.sync_time = sync_time

# Set default
Log.setLevel(logging.INFO)
Log.propagate = True
