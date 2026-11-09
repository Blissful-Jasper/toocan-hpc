"""Parameter-file parser (same `key = value` syntax as the original TOOCAN .dat files)."""
from collections import defaultdict


def parse_param_file(path):
    d = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";") or "=" not in line:
                continue
            k, v = (s.strip() for s in line.split("=", 1))
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    v = v.strip('"').strip("'")
            d[k].append(v)
    return {k: (v[0] if len(v) == 1 else v) for k, v in d.items()}
