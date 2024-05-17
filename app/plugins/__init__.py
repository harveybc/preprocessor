# Initialize the plugins package
import pkg_resources

# Register all available plugins in the entry points for dynamic loading
pkg_resources.declare_namespace(__name__)
