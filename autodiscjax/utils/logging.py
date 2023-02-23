from autodiscjax import DictTree

def append_to_log(logs, log_data):
    if log_data is None:
        pass
    elif isinstance(log_data, DictTree):
        for k, v in log_data.items():
            if k not in logs:
                logs[k] = [v]
            else:
                logs[k].append(v)
    else:
        raise NotImplementedError

    return logs