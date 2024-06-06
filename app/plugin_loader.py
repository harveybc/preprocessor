import importlib

def load_plugin(plugin_name):
    try:
        module_name = f'app.plugins.plugin_{plugin_name}'
        plugin_module = importlib.import_module(module_name)
        plugin_class = plugin_module.Plugin
        required_params = list(plugin_class.plugin_params.keys())
        return plugin_class(), required_params
    except ImportError as e:
        print(f"Failed to get plugin params: {plugin_name}, Error: {e}")
        return None, []

def get_plugin_params(plugin_name):
    try:
        module_name = f'app.plugins.plugin_{plugin_name}'
        plugin_module = importlib.import_module(module_name)
        plugin_class = plugin_module.Plugin
        return list(plugin_class.plugin_params.keys())
    except ImportError as e:
        print(f"Failed to get plugin params: {plugin_name}, Error: {e}")
        return []

def add_plugin_params(parser, plugin_name):
    plugin_params = get_plugin_params(plugin_name)
    for param in plugin_params:
        parser.add_argument(f'--{param}', type=str, required=False)
