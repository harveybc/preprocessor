#!/usr/bin/env python3

# Simple test to verify dependency resolution fix
from app.core.plugin_loader import PluginLoader, PluginMetadata

loader = PluginLoader()

# Add test plugins
base = PluginMetadata('base_plugin', '1.0.0', 'Base plugin', 'Author', 'base')
dependent = PluginMetadata('dependent_plugin', '1.0.0', 'Dependent plugin', 'Author', 'test', dependencies=['base_plugin'])

loader.discovered_plugins['base_plugin'] = base
loader.discovered_plugins['dependent_plugin'] = dependent

# Test dependency resolution
if loader.build_dependency_graph():
    order = loader.get_loading_order()
    if len(order) == 2 and order.index('base_plugin') < order.index('dependent_plugin'):
        print("PASS: Dependency resolution working correctly")
        exit(0)
    else:
        print(f"FAIL: Incorrect loading order: {order}")
        exit(1)
else:
    print("FAIL: Dependency graph building failed")
    exit(1)
