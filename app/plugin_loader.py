import pkg_resources

def load_plugin(plugin_name):
    try:
        entry_point = next(pkg_resources.iter_entry_points('preprocessor.plugins', plugin_name))
        plugin = entry_point.load()
        required_params = getattr(plugin, 'plugin_params', [])
        return plugin(), required_params
    except StopIteration:
        print(f"Plugin {plugin_name} not found.", file=sys.stderr)
        return None, []