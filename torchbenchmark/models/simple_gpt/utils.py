import os

# provided by torchrun
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
