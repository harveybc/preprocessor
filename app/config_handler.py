import json

def load_config(args):
    if args.load_config:
        with open(args.load_config, 'r') as f:
            return json.load(f)
    return {}

def save_config(config, path='config_out.json'):
    with open(path, 'w') as f:
        json.dump(config, f)
    return config, path

def merge_config(config, cli_args):
    config.update({k: v for k, v in cli_args.items() if v is not None})
    return config

def save_debug_info(debug_info, path='debug_out.json'):
    with open(path, 'w') as f:
        json.dump(debug_info, f)
