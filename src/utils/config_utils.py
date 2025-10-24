import yaml, copy

def _apply_overrides(cfg, overrides):
    for o in overrides:
        if "=" not in o: 
            continue
        k, v = o.split("=", 1)
        keys = k.split(".")
        cur = cfg
        for kk in keys[:-1]:
            if kk not in cur or not isinstance(cur[kk], dict):
                cur[kk] = {}
            cur = cur[kk]
        try:
            vv = yaml.safe_load(v)
        except Exception:
            vv = v
        cur[keys[-1]] = vv
    return cfg

def load_config(path, overrides=None):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = _apply_overrides(cfg, overrides or [])
    return cfg

def normalize_config(cfg: dict) -> dict:
    def sort_dict(d):
        if isinstance(d, dict):
            return {k: sort_dict(d[k]) for k in sorted(d.keys())}
        elif isinstance(d, list):
            return [sort_dict(x) for x in d]
        else:
            return d
    return sort_dict(copy.deepcopy(cfg))