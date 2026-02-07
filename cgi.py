 

def parse_header(header_value: str):
    """Parse a header like 'text/plain; charset=utf-8' into (value, params dict).

    This is intentionally minimal and only supports simple key=value params.
    """
    if not header_value:
        return "", {}

    parts = [p.strip() for p in header_value.split(';')]
    main_value = parts[0]
    params = {}
    for part in parts[1:]:
        if '=' in part:
            k, v = part.split('=', 1)
            params[k.strip()] = v.strip().strip('"')
    return main_value, params
