import importlib

def load_plugin(plugin_name):
    try:
        print(f"Loading plugin: {plugin_name}")
        plugin_module = importlib.import_module(f'app.plugins.plugin_{plugin_name}')
        plugin_class = getattr(plugin_module, 'Plugin')
        required_params = list(plugin_class.plugin_params.keys())
        print(f"Successfully loaded plugin: {plugin_name} with params: {required_params}")
        return plugin_class(), required_params
    except Exception as e:
        print(f"Failed to load plugin: {plugin_name}, Error: {e}")
        return None, []

def get_plugin_params(plugin_name):
    try:
        plugin_module = importlib.import_module(f'app.plugins.plugin_{plugin_name}')
        plugin_class = getattr(plugin_module, 'Plugin')
        return plugin_class.plugin_params
    except Exception as e:
        print(f"Failed to get plugin params: {plugin_name}, Error: {e}")
        return {}
