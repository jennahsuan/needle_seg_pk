from copy import deepcopy

def merge(config, new_config):
    for k in new_config.keys():
        if k in config.keys():
            config[k].update(new_config[k])
        else:
            config[k]=new_config[k]
    return config