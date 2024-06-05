import pkg_resources

def load_plugin(plugin_name):
    """
    Load the specified plugin and retrieve its required parameters.

    Args:
        plugin_name (str): The name of the plugin to load.

    Returns:
        tuple: The loaded plugin instance and a list of required parameters.
    """
    try:
        entry_point = next(pkg_resources.iter_entry_points('preprocessor.plugins', plugin_name))
        plugin = entry_point.load()
        required_params = list(plugin.plugin_params.keys())
        print(f"Loaded plugin: {plugin_name} with params: {required_params}")
        return plugin(), required_params
    except StopIteration:
        print(f"Plugin {plugin_name} not found.", file=sys.stderr)
        return None, []

def set_plugin_params(plugin, config, required_params):
    """
    Set the parameters for the plugin based on the provided configuration.

    Args:
        plugin: The plugin instance.
        config (dict): The configuration dictionary.
        required_params (list): The list of parameters required by the plugin.

    Returns:
        None
    """
    plugin_params = {param: config.get(param, plugin.plugin_params[param]) for param in required_params}
    print(f"Setting plugin params: {plugin_params}")
    plugin.set_params(**plugin_params)
