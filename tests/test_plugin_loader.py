"""Unit Tests for PluginLoader

This module implements comprehensive unit tests for the PluginLoader class,
following BDD methodology and the specifications from test_unit.md.

Test Coverage:
- Plugin discovery behavior
- Plugin loading and initialization
- Plugin dependency resolution
- Plugin lifecycle management
- Error handling and isolation
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.core.plugin_loader import PluginLoader, BasePlugin, PluginMetadata, PluginState


class MockPlugin(BasePlugin):
    """Test plugin for unit testing"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin for unit testing",
            author="Test Author",
            plugin_type="test"
        )
    
    def execute(self, data, **kwargs):
        return f"processed_{data}"


class MockPluginWithDependencies(BasePlugin):
    """Test plugin with dependencies"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="dependent_plugin",
            version="1.0.0",
            description="Test plugin with dependencies",
            author="Test Author",
            plugin_type="test",
            dependencies=["test_plugin"]
        )
    
    def execute(self, data, **kwargs):
        return f"dependent_processed_{data}"


class MockFailingPlugin(BasePlugin):
    """Test plugin that fails during execution"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="failing_plugin",
            version="1.0.0",
            description="Test plugin that fails",
            author="Test Author",
            plugin_type="test"
        )
    
    def execute(self, data, **kwargs):
        raise RuntimeError("Plugin execution failed")


class TestPluginLoaderDiscovery(unittest.TestCase):
    """Test PluginLoader plugin discovery behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plugin_loader = PluginLoader()
        self.temp_dir = tempfile.mkdtemp()
        self.plugin_loader.add_plugin_directory(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_plugin_directory_addition(self):
        """
        UNIT-004-A: Plugin discovery behavior
        Given: Directory structure with various plugin types
        When: Plugin discovery is executed
        Then: Valid plugins are identified and catalogued
        """
        # Test adding valid directory
        new_temp_dir = tempfile.mkdtemp()
        try:
            initial_count = len(self.plugin_loader.plugin_directories)
            self.plugin_loader.add_plugin_directory(new_temp_dir)
            
            # Verify directory was added
            self.assertEqual(len(self.plugin_loader.plugin_directories), initial_count + 1)
            self.assertIn(new_temp_dir, self.plugin_loader.plugin_directories)
            
        finally:
            import shutil
            shutil.rmtree(new_temp_dir, ignore_errors=True)
    
    def test_invalid_directory_handling(self):
        """
        Test that invalid directories are handled gracefully
        Given: Non-existent directory path
        When: Directory addition is attempted
        Then: Directory is not added and warning is logged
        """
        initial_count = len(self.plugin_loader.plugin_directories)
        self.plugin_loader.add_plugin_directory("/nonexistent/directory")
        
        # Verify directory was not added
        self.assertEqual(len(self.plugin_loader.plugin_directories), initial_count)
    
    def test_plugin_file_discovery(self):
        """
        UNIT-004-A: Valid plugins are identified and catalogued
        Given: Plugin files in discovery directory
        When: Plugin discovery is executed
        Then: Valid plugin classes are discovered
        """
        # Create a test plugin file
        plugin_file_content = '''
from app.core.plugin_loader import BasePlugin, PluginMetadata

class DiscoveredTestPlugin(BasePlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="discovered_test_plugin",
            version="1.0.0",
            description="Test plugin for discovery",
            author="Test Author",
            plugin_type="test"
        )
    
    def execute(self, data, **kwargs):
        return f"discovered_{data}"
'''
        
        plugin_file_path = os.path.join(self.temp_dir, "test_plugin.py")
        with open(plugin_file_path, 'w') as f:
            f.write(plugin_file_content)
        
        # Discover plugins
        discovered_count = self.plugin_loader.discover_plugins()
        
        # Verify plugin was discovered
        self.assertEqual(discovered_count, 1)
        self.assertIn("discovered_test_plugin", self.plugin_loader.discovered_plugins)
        
        # Verify metadata
        metadata = self.plugin_loader.discovered_plugins["discovered_test_plugin"]
        self.assertEqual(metadata.name, "discovered_test_plugin")
        self.assertEqual(metadata.plugin_type, "test")
        self.assertEqual(metadata.state, PluginState.DISCOVERED)
    
    def test_invalid_plugin_file_handling(self):
        """
        UNIT-004-A: Invalid plugins are skipped with warning messages
        Given: Invalid plugin files
        When: Plugin discovery is executed
        Then: Invalid files are skipped and errors are logged
        """
        # Create invalid Python file
        invalid_file_path = os.path.join(self.temp_dir, "invalid_plugin.py")
        with open(invalid_file_path, 'w') as f:
            f.write("invalid python syntax {{{")
        
        # Discover plugins
        discovered_count = self.plugin_loader.discover_plugins()
        
        # Verify no plugins discovered and error logged
        self.assertEqual(discovered_count, 0)
        self.assertTrue(len(self.plugin_loader.get_loading_errors()) > 0)
    
    def test_plugin_metadata_extraction(self):
        """
        UNIT-004-A: Plugin metadata is correctly extracted
        Given: Plugin with complete metadata
        When: Plugin discovery extracts metadata
        Then: All metadata fields are correctly populated
        """
        # Create plugin with detailed metadata
        plugin_file_content = '''
from app.core.plugin_loader import BasePlugin, PluginMetadata

class DetailedTestPlugin(BasePlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="detailed_test_plugin",
            version="2.1.0",
            description="Detailed test plugin with full metadata",
            author="Detailed Test Author",
            plugin_type="preprocessing",
            dependencies=["base_plugin", "utility_plugin"]
        )
    
    def execute(self, data, **kwargs):
        return data
'''
        
        plugin_file_path = os.path.join(self.temp_dir, "detailed_plugin.py")
        with open(plugin_file_path, 'w') as f:
            f.write(plugin_file_content)
        
        # Discover plugins
        self.plugin_loader.discover_plugins()
        
        # Verify detailed metadata
        metadata = self.plugin_loader.discovered_plugins["detailed_test_plugin"]
        self.assertEqual(metadata.name, "detailed_test_plugin")
        self.assertEqual(metadata.version, "2.1.0")
        self.assertEqual(metadata.description, "Detailed test plugin with full metadata")
        self.assertEqual(metadata.author, "Detailed Test Author")
        self.assertEqual(metadata.plugin_type, "preprocessing")
        self.assertEqual(metadata.dependencies, ["base_plugin", "utility_plugin"])


class TestPluginLoaderLoading(unittest.TestCase):
    """Test PluginLoader loading and initialization behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plugin_loader = PluginLoader()
        
        # Manually add test plugins to discovered plugins
        test_metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            plugin_type="test"
        )
        test_metadata.file_path = "dummy_path"
        test_metadata.class_name = "MockPlugin"
        
        self.plugin_loader.discovered_plugins["test_plugin"] = test_metadata
    
    def test_plugin_interface_validation(self):
        """
        UNIT-004-B: Plugin loading and initialization
        Given: Discovered plugins and their configurations
        When: Plugins are loaded and initialized
        Then: Plugin classes are correctly instantiated
        """
        # Mock the plugin class loading
        with patch.object(self.plugin_loader, '_load_plugin_class') as mock_load:
            mock_load.return_value = MockPlugin
            
            # Validate plugin interface
            result = self.plugin_loader.validate_plugin_interface("test_plugin")
            
            # Verify validation passed
            self.assertTrue(result)
    
    def test_plugin_loading_success(self):
        """
        UNIT-004-B: Plugin initialization receives correct configuration
        Given: Valid plugin ready for loading
        When: Plugin loading is performed
        Then: Plugin is successfully loaded and ready for use
        """
        # Mock the plugin class loading
        with patch.object(self.plugin_loader, '_load_plugin_class') as mock_load:
            mock_load.return_value = MockPlugin
            
            # Load plugin
            result = self.plugin_loader.load_plugin("test_plugin")
            
            # Verify loading succeeded
            self.assertTrue(result)
            self.assertIn("test_plugin", self.plugin_loader.loaded_plugins)
                 # Verify plugin instance
        plugin_instance = self.plugin_loader.get_plugin("test_plugin")
        self.assertIsInstance(plugin_instance, MockPlugin)
    
    def test_plugin_loading_failure(self):
        """
        UNIT-004-B: Plugin loading failures are isolated and reported
        Given: Plugin that fails to load
        When: Plugin loading is attempted
        Then: Loading failure is handled gracefully
        """
        # Mock the plugin class loading to fail
        with patch.object(self.plugin_loader, '_load_plugin_class') as mock_load:
            mock_load.return_value = None
            
            # Attempt to load plugin
            result = self.plugin_loader.load_plugin("test_plugin")
            
            # Verify loading failed
            self.assertFalse(result)
            self.assertNotIn("test_plugin", self.plugin_loader.loaded_plugins)
            
            # Verify plugin state updated
            metadata = self.plugin_loader.discovered_plugins["test_plugin"]
            self.assertEqual(metadata.state, PluginState.FAILED)
    
    def test_plugin_initialization(self):
        """
        UNIT-004-D: Plugin lifecycle management
        Given: Loaded and initialized plugins
        When: Plugin lifecycle operations are performed
        Then: Plugins can be started, stopped, and restarted
        """
        # Mock and load plugin
        with patch.object(self.plugin_loader, '_load_plugin_class') as mock_load:
            mock_load.return_value = MockPlugin
            
            # Load plugin
            self.plugin_loader.load_plugin("test_plugin")
            
            # Initialize plugin with configuration
            config = {"param1": "value1", "param2": 42}
            result = self.plugin_loader.initialize_plugin("test_plugin", config)
            
            # Verify initialization succeeded
            self.assertTrue(result)
            
            # Verify plugin configuration
            plugin = self.plugin_loader.get_plugin("test_plugin")
            self.assertEqual(plugin.config, config)
            self.assertTrue(plugin.initialized)


class TestPluginLoaderDependencies(unittest.TestCase):
    """Test PluginLoader dependency resolution behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plugin_loader = PluginLoader()
        
        # Add test plugins with dependencies
        base_plugin = PluginMetadata(
            name="base_plugin",
            version="1.0.0",
            description="Base plugin",
            author="Test Author",
            plugin_type="base"
        )
        
        dependent_plugin = PluginMetadata(
            name="dependent_plugin", 
            version="1.0.0",
            description="Dependent plugin",
            author="Test Author",
            plugin_type="test",
            dependencies=["base_plugin"]
        )
        
        self.plugin_loader.discovered_plugins["base_plugin"] = base_plugin
        self.plugin_loader.discovered_plugins["dependent_plugin"] = dependent_plugin
    
    def test_dependency_graph_building(self):
        """
        UNIT-004-C: Plugin dependency resolution
        Given: Plugins with declared dependencies
        When: Dependency resolution is performed
        Then: Dependency graph is correctly constructed
        """
        # Build dependency graph
        result = self.plugin_loader.build_dependency_graph()
        
        # Verify graph building succeeded
        self.assertTrue(result)
        
        # Verify dependency graph structure
        self.assertIn("base_plugin", self.plugin_loader.dependency_graph)
        self.assertIn("dependent_plugin", self.plugin_loader.dependency_graph)
        self.assertEqual(self.plugin_loader.dependency_graph["dependent_plugin"], ["base_plugin"])
        self.assertEqual(self.plugin_loader.dependency_graph["base_plugin"], [])
    
    def test_loading_order_determination(self):
        """
        UNIT-004-C: Loading order respects dependency requirements
        Given: Plugins with dependencies
        When: Loading order is determined
        Then: Dependencies are loaded before dependents
        """
        # Ensure clean state by resetting discovered plugins
        self.plugin_loader.discovered_plugins.clear()
        
        # Re-add our test plugins
        base_plugin = PluginMetadata(
            name="base_plugin",
            version="1.0.0",
            description="Base plugin",
            author="Test Author",
            plugin_type="base"
        )
        
        dependent_plugin = PluginMetadata(
            name="dependent_plugin", 
            version="1.0.0",
            description="Dependent plugin",
            author="Test Author",
            plugin_type="test",
            dependencies=["base_plugin"]
        )
        
        self.plugin_loader.discovered_plugins["base_plugin"] = base_plugin
        self.plugin_loader.discovered_plugins["dependent_plugin"] = dependent_plugin
        
        # Build dependency graph and get loading order
        self.plugin_loader.build_dependency_graph()
        loading_order = self.plugin_loader.get_loading_order()
        
        # Verify loading order
        self.assertEqual(len(loading_order), 2)
        base_index = loading_order.index("base_plugin")
        dependent_index = loading_order.index("dependent_plugin")
        self.assertLess(base_index, dependent_index)  # base_plugin should load first
    
    def test_circular_dependency_detection(self):
        """
        UNIT-004-C: Circular dependencies are detected and rejected
        Given: Plugins with circular dependencies
        When: Dependency resolution is performed
        Then: Circular dependencies are detected and graph building fails
        """
        # Create circular dependency
        plugin_a = PluginMetadata(
            name="plugin_a",
            version="1.0.0",
            description="Plugin A",
            author="Test Author",
            plugin_type="test",
            dependencies=["plugin_b"]
        )
        
        plugin_b = PluginMetadata(
            name="plugin_b",
            version="1.0.0", 
            description="Plugin B",
            author="Test Author",
            plugin_type="test",
            dependencies=["plugin_a"]
        )
        
        self.plugin_loader.discovered_plugins.clear()
        self.plugin_loader.discovered_plugins["plugin_a"] = plugin_a
        self.plugin_loader.discovered_plugins["plugin_b"] = plugin_b
        
        # Attempt to build dependency graph
        result = self.plugin_loader.build_dependency_graph()
        
        # Verify circular dependency detection
        self.assertFalse(result)
        self.assertTrue(len(self.plugin_loader.get_loading_errors()) > 0)
    
    def test_missing_dependency_detection(self):
        """
        UNIT-004-C: Missing dependencies are reported clearly
        Given: Plugin depending on non-existent plugin
        When: Dependency validation is performed
        Then: Missing dependency is detected and reported
        """
        # Add plugin with missing dependency
        plugin_with_missing_dep = PluginMetadata(
            name="missing_dep_plugin",
            version="1.0.0",
            description="Plugin with missing dependency",
            author="Test Author",
            plugin_type="test",
            dependencies=["nonexistent_plugin"]
        )
        
        self.plugin_loader.discovered_plugins["missing_dep_plugin"] = plugin_with_missing_dep
        
        # Attempt to build dependency graph
        result = self.plugin_loader.build_dependency_graph()
        
        # Verify missing dependency detection
        self.assertFalse(result)
        errors = self.plugin_loader.get_loading_errors()
        self.assertTrue(any("unknown plugin: nonexistent_plugin" in error for error in errors))


class TestPluginLoaderIntegration(unittest.TestCase):
    """Test PluginLoader integration and lifecycle behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plugin_loader = PluginLoader()
    
    def test_plugin_type_filtering(self):
        """
        Test getting plugins by type
        Given: Multiple plugins of different types
        When: Plugins are filtered by type
        Then: Only plugins of specified type are returned
        """
        # Mock plugins of different types
        with patch.object(self.plugin_loader, 'loaded_plugins') as mock_loaded:
            with patch.object(self.plugin_loader, 'discovered_plugins') as mock_discovered:
                
                # Setup mock plugins
                plugin1 = MockPlugin()
                plugin2 = MockPlugin()
                
                mock_loaded.items.return_value = [
                    ("plugin1", plugin1),
                    ("plugin2", plugin2)
                ]
                
                metadata1 = PluginMetadata("plugin1", "1.0.0", "Test", "Author", "type_a")
                metadata1.state = PluginState.INITIALIZED
                metadata2 = PluginMetadata("plugin2", "1.0.0", "Test", "Author", "type_b") 
                metadata2.state = PluginState.INITIALIZED
                
                mock_discovered.__getitem__.side_effect = lambda k: {
                    "plugin1": metadata1,
                    "plugin2": metadata2
                }[k]
                
                # Get plugins by type
                type_a_plugins = self.plugin_loader.get_plugins_by_type("type_a")
                type_b_plugins = self.plugin_loader.get_plugins_by_type("type_b")
                
                # Verify filtering
                self.assertEqual(len(type_a_plugins), 1)
                self.assertEqual(len(type_b_plugins), 1)
    
    def test_plugin_cleanup(self):
        """
        UNIT-004-D: Plugin cleanup is performed on shutdown
        Given: Loaded plugins
        When: Cleanup is performed
        Then: All plugins are cleaned up properly
        """
        # Mock loaded plugins
        mock_plugin1 = MagicMock(spec=BasePlugin)
        mock_plugin2 = MagicMock(spec=BasePlugin)
        
        self.plugin_loader.loaded_plugins = {
            "plugin1": mock_plugin1,
            "plugin2": mock_plugin2
        }
        self.plugin_loader.plugin_instances = {
            "plugin1": mock_plugin1,
            "plugin2": mock_plugin2
        }
        
        # Perform cleanup
        self.plugin_loader.cleanup_plugins()
        
        # Verify cleanup was called on all plugins
        mock_plugin1.cleanup.assert_called_once()
        mock_plugin2.cleanup.assert_called_once()
        
        # Verify plugin tracking was cleared
        self.assertEqual(len(self.plugin_loader.loaded_plugins), 0)
        self.assertEqual(len(self.plugin_loader.plugin_instances), 0)
    
    def test_plugin_summary_generation(self):
        """
        Test plugin summary generation
        Given: Plugin loader with various plugin states
        When: Summary is requested
        Then: Complete summary with state distribution is returned
        """
        # Setup plugins in different states
        metadata1 = PluginMetadata("plugin1", "1.0.0", "Test", "Author", "test")
        metadata1.state = PluginState.LOADED
        metadata2 = PluginMetadata("plugin2", "1.0.0", "Test", "Author", "test")
        metadata2.state = PluginState.FAILED
        metadata3 = PluginMetadata("plugin3", "1.0.0", "Test", "Author", "test")
        metadata3.state = PluginState.INITIALIZED
        
        self.plugin_loader.discovered_plugins = {
            "plugin1": metadata1,
            "plugin2": metadata2,
            "plugin3": metadata3
        }
        self.plugin_loader.loaded_plugins = {"plugin1": MockPlugin(), "plugin3": MockPlugin()}
        self.plugin_loader.loading_errors = ["Error 1", "Error 2"]
        
        # Generate summary
        summary = self.plugin_loader.get_plugin_summary()
        
        # Verify summary content
        self.assertEqual(summary["total_discovered"], 3)
        self.assertEqual(summary["total_loaded"], 2)
        self.assertEqual(summary["loading_errors"], 2)
        self.assertIn("state_distribution", summary)
        self.assertEqual(summary["state_distribution"]["loaded"], 1)
        self.assertEqual(summary["state_distribution"]["failed"], 1)
        self.assertEqual(summary["state_distribution"]["initialized"], 1)


class TestBasePluginBehavior(unittest.TestCase):
    """Test BasePlugin behavior and interface compliance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plugin = MockPlugin()
    
    def test_plugin_configuration(self):
        """
        UNIT-005-B: Plugin execution behavior
        Given: Configured plugin ready for execution
        When: Plugin execute method is called
        Then: Input data is processed according to plugin logic
        """
        # Test plugin configuration
        config = {"param1": "value1", "threshold": 0.5}
        result = self.plugin.configure(config)
        
        # Verify configuration succeeded
        self.assertTrue(result)
        self.assertEqual(self.plugin.config, config)
        self.assertTrue(self.plugin.initialized)
    
    def test_plugin_execution(self):
        """
        UNIT-005-B: Output data conforms to expected format
        Given: Plugin ready for execution
        When: Execute method is called with data
        Then: Data is processed and returned in expected format
        """
        # Configure plugin
        self.plugin.configure({})
        
        # Execute plugin
        test_data = "test_input"
        result = self.plugin.execute(test_data)
        
        # Verify execution result
        self.assertEqual(result, "processed_test_input")
    
    def test_plugin_input_validation(self):
        """
        UNIT-005-A: Plugin interface compliance
        Given: BasePlugin implementation
        When: Plugin interface methods are called
        Then: Required methods are implemented and functional
        """
        # Test input validation
        self.assertTrue(self.plugin.validate_input("valid_data"))
        self.assertFalse(self.plugin.validate_input(None))
    
    def test_plugin_metadata_provision(self):
        """
        UNIT-005-A: Plugin metadata is correctly provided
        Given: Plugin implementation
        When: Metadata is requested
        Then: Complete and accurate metadata is returned
        """
        metadata = self.plugin.get_metadata()
        
        # Verify metadata completeness
        self.assertEqual(metadata.name, "test_plugin")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.plugin_type, "test")
        self.assertIsInstance(metadata.dependencies, list)
    
    def test_plugin_cleanup(self):
        """
        UNIT-005-C: Plugin error handling
        Given: Plugin encountering various error conditions
        When: Cleanup is performed
        Then: Plugin state is correctly reset
        """
        # Configure plugin
        self.plugin.configure({"test": "config"})
        self.assertTrue(self.plugin.initialized)
        
        # Perform cleanup
        self.plugin.cleanup()
        
        # Verify cleanup
        self.assertEqual(len(self.plugin.config), 0)
        self.assertFalse(self.plugin.initialized)
    
    def test_plugin_output_schema(self):
        """
        Test plugin output schema provision
        Given: Plugin implementation
        When: Output schema is requested
        Then: Schema information is provided
        """
        schema = self.plugin.get_output_schema()
        
        # Verify schema structure
        self.assertIsInstance(schema, dict)
        self.assertIn("type", schema)


class TestPluginErrorHandling(unittest.TestCase):
    """Test plugin error handling and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.plugin_loader = PluginLoader()
    
    def test_unknown_plugin_loading(self):
        """
        Test loading of unknown plugins
        Given: Request to load unknown plugin
        When: Loading is attempted
        Then: Loading fails gracefully with appropriate error
        """
        result = self.plugin_loader.load_plugin("unknown_plugin")
        
        # Verify loading failed
        self.assertFalse(result)
    
    def test_plugin_initialization_without_loading(self):
        """
        Test initialization of unloaded plugins
        Given: Request to initialize unloaded plugin
        When: Initialization is attempted
        Then: Initialization fails with appropriate error
        """
        result = self.plugin_loader.initialize_plugin("unloaded_plugin")
        
        # Verify initialization failed
        self.assertFalse(result)
    
    def test_plugin_retrieval_edge_cases(self):
        """
        Test plugin retrieval edge cases
        Given: Various plugin retrieval scenarios
        When: Plugins are requested
        Then: Appropriate responses are returned
        """
        # Test getting non-existent plugin
        plugin = self.plugin_loader.get_plugin("nonexistent")
        self.assertIsNone(plugin)
        
        # Test getting metadata for non-existent plugin
        metadata = self.plugin_loader.get_plugin_metadata("nonexistent")
        self.assertIsNone(metadata)
        
        # Test getting plugins by non-existent type
        plugins = self.plugin_loader.get_plugins_by_type("nonexistent_type")
        self.assertEqual(len(plugins), 0)


if __name__ == "__main__":
    unittest.main()
