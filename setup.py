from setuptools import setup, find_packages

setup(
    name='preprocessor',
    version='0.1.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'preprocessor=app.main:main'
        ],
        'preprocessor.plugins': [
            'default_plugin=app.plugins.plugin_default:Plugin',
            'normalizer=app.plugins.plugin_normalizer:Plugin',
            'unbiaser=app.plugins.plugin_unbiaser:Plugin',
            'trimmer=app.plugins.plugin_trimmer:Plugin',
            'feature_selector=app.plugins.plugin_feature_selector_pre:Plugin',
            'cleaner=app.plugins.plugin_cleaner:Plugin'
        ]
    },
    install_requires=[
        'pandas',
        'numpy',
        'requests'
    ],
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description='A preprocessing system that supports dynamic loading of plugins for processing CSV data.'
)
