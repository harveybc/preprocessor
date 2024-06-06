import pytest
import pandas as pd
from app.plugin_loader import load_plugin

def test_unbiaser_plugin_integration():
    plugin_class, required_params = load_plugin('unbiaser')
    plugin = plugin_class()
    data = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6], 'col2': [6, 5, 4, 3, 2, 1]})
    processed_data = plugin.process(data)
    assert not processed_data.empty
