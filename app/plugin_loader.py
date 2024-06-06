import importlib
import sys
import os

def load_plugin(plugin_name):
    print(f"Attempting to load plugin: {plugin_name}")
    try:
        plugin_module = importlib.import_module(f'app.plugins.plugin_{plugin_name}')
        plugin_class = plugin_module.Plugin
        required_params = list(plugin_class.plugin_params.keys())
        print(f"Successfully loaded plugin: {plugin_name} with params: {required_params}")
        return plugin_class, required_params
    except ImportError as e:
        print(f"Failed to load plugin: {plugin_name}, Error: {e}")
        print(f"Current sys.path: {sys.path}")
        plugin_path = os.path.join(os.path.dirname(__file__), 'plugins', f'plugin_{plugin_name}.py')
        print(f"Expected plugin path: {plugin_path}")
        if not os.path.exists(plugin_path):
            print(f"Plugin file not found at: {plugin_path}")
        return None, []
    except AttributeError as e:
        print(f"Plugin {plugin_name} does not define required attributes, Error: {e}")
        return None, []

def get_plugin_params(plugin_name):
    print(f"Getting plugin parameters for: {plugin_name}")
    try:
        plugin_module = importlib.import_module(f'app.plugins.plugin_{plugin_name}')
        plugin_class = plugin_module.Plugin
        plugin_params = plugin_class.plugin_params
        print(f"Retrieved plugin parameters for {plugin_name}: {plugin_params}")
        return plugin_params
    except ImportError as e:
        print(f"Failed to import plugin {plugin_name}, Error: {e}")
        print(f"Current sys.path: {sys.path}")
        plugin_path = os.path.join(os.path.dirname(__file__), 'plugins', f'plugin_{plugin_name}.py')
        print(f"Expected plugin path: {plugin_path}")
        if not os.path.exists(plugin_path):
            print(f"Plugin file not found at: {plugin_path}")
        return {}
    except AttributeError as e:
        print(f"Plugin {plugin_name} does not define parameters, Error: {e}")
        return {}
