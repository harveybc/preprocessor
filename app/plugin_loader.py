import pkg_resources

def load_plugin(plugin_name):
    print(f"Loading plugin: {plugin_name}")  # Debug message
    try:
        entry_point = next(pkg_resources.iter_entry_points('preprocessor.plugins', plugin_name))
        plugin_class = entry_point.load()
        print(f"Successfully loaded plugin: {plugin_name}")  # Debug message
        required_params = list(plugin_class.plugin_params.keys())
        return plugin_class, required_params
    except StopIteration:
        print(f"Plugin {plugin_name} not found, loading DefaultPlugin.")  # Debug message
        from app.default_plugin import DefaultPlugin
        return DefaultPlugin, []

def load_plugin_fallback(plugin_name):
    print(f"Falling back to load DefaultPlugin for plugin: {plugin_name}")  # Debug message
    from app.default_plugin import DefaultPlugin
    return DefaultPlugin, []

# Modify the main.py to include additional debug messages.
