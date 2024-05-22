from .Dump import dump_util
from .hijack_tool import hijack_target_api
from . import config

cfg = config.cfg


class Acc:
    def __init__(self):
        # Hijack target api at __init__.
        hijack_target_api()

    def start(self):
        # global step counting.
        cfg.new_step()

    def stop(self):
        if cfg.dump_state:
            dump_util.dump()
