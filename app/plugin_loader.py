import pkg_resources
import sys

def load_plugin(plugin_name):
    print(f"Attempting to load plugin: {plugin_name}")
    try:
        entry_point_map = pkg_resources.get_entry_map('preprocessor', 'preprocessor.plugins')
        print(f"Available entry points: {entry_point_map.keys()}")
        entry_point = entry_point_map[plugin_name]
        plugin_class = entry_point.load()
        required_params = list(plugin_class.plugin_params.keys())
        print(f"Successfully loaded plugin: {plugin_name} with params: {plugin_class.plugin_params}")
        return plugin_class, required_params
    except KeyError as e:
        print(f"Failed to find plugin {plugin_name}, Error: {e}")
        raise ImportError(f"Plugin {plugin_name} not found.")
    except Exception as e:
        print(f"Failed to load plugin {plugin_name}, Error: {e}")
        raise

def get_plugin_params(plugin_name):
    print(f"Getting plugin parameters for: {plugin_name}")
    try:
        entry_point_map = pkg_resources.get_entry_map('preprocessor', 'preprocessor.plugins')
        print(f"Available entry points: {entry_point_map.keys()}")
        entry_point = entry_point_map[plugin_name]
        plugin_class = entry_point.load()
        print(f"Retrieved plugin params: {plugin_class.plugin_params}")
        return plugin_class.plugin_params
    except KeyError as e:
        print(f"Failed to find plugin {plugin_name}, Error: {e}")
        return {}
    except Exception as e:
        print(f"Failed to get plugin params: {plugin_name}, Error: {e}")
        return {}
