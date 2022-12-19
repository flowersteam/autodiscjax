from autodiscjax import DictTree
import exputils.data.logging as log

def append_to_log(log_data):
    if log_data is None:
        return
    elif isinstance(log_data, DictTree):
        for k, v in log_data.items():
            log.add_value(k, v)
    else:
        raise NotImplementedError