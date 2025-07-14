"""Command Line Interface for Preprocessor System

This module implements the CLI interface for the preprocessor system,
providing a clean command-line interface that orchestrates the core
preprocessing pipeline.

Following BDD design from design_system.md and test_system.md.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from app.core.configuration_manager import ConfigurationManager
from app.core.preprocessor_core import PreprocessorCore


class CLIInterface:
    """Command Line Interface for the preprocessor system"""
    
    def __init__(self):
        """Initialize the CLI interface"""
        self.config_manager = None
        self.preprocessor_core = None
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser"""
        parser = argparse.ArgumentParser(
            description='Preprocessor System - Data preprocessing for time series prediction',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic preprocessing with default configuration
  python -m app.cli input_data.csv
  
  # Preprocessing with custom configuration
  python -m app.cli input_data.csv --config config.json
  
  # Preprocessing with output directory
  python -m app.cli input_data.csv --output-dir ./output
  
  # Preprocessing with feature engineering plugins
  python -m app.cli input_data.csv --feature-plugins pca,polynomial
  
  # Preprocessing with custom split ratios
  python -m app.cli input_data.csv --split-ratios 0.6,0.15,0.15,0.05,0.025,0.025
            """
        )
        
        # Positional arguments
        parser.add_argument(
            'input_file',
            help='Path to input data file (CSV, JSON, or Parquet)'
        )
        
        # Configuration options
        parser.add_argument(
            '--config', '-c',
            type=str,
            help='Path to configuration file (JSON)'
        )
        
        parser.add_argument(
            '--output-dir', '-o',
            type=str,
            default='./output',
            help='Output directory for processed datasets (default: ./output)'
        )
        
        # Data splitting options
        parser.add_argument(
            '--split-ratios',
            type=str,
            help='Dataset split ratios as comma-separated values (6 values, sum=1.0)'
        )
        
        parser.add_argument(
            '--split-method',
            choices=['temporal', 'random', 'stratified'],
            default='temporal',
            help='Data splitting method (default: temporal)'
        )
        
        # Normalization options
        parser.add_argument(
            '--normalization-method',
            choices=['z-score', 'min-max', 'robust', 'none'],
            default='z-score',
            help='Normalization method (default: z-score)'
        )
        
        parser.add_argument(
            '--training-datasets',
            type=str,
            default='d1,d2',
            help='Datasets used for normalization training (default: d1,d2)'
        )
        
        # Plugin options
        parser.add_argument(
            '--feature-plugins',
            type=str,
            help='Feature engineering plugins as comma-separated list'
        )
        
        parser.add_argument(
            '--postprocessing-plugins',
            type=str,
            help='Postprocessing plugins as comma-separated list'
        )
        
        parser.add_argument(
            '--plugin-dirs',
            type=str,
            help='Additional plugin directories as comma-separated paths'
        )
        
        # Output options
        parser.add_argument(
            '--output-format',
            choices=['csv', 'json', 'parquet'],
            default='csv',
            help='Output file format (default: csv)'
        )
        
        parser.add_argument(
            '--save-config',
            type=str,
            help='Save effective configuration to specified file'
        )
        
        parser.add_argument(
            '--save-metadata',
            action='store_true',
            help='Save processing metadata with output'
        )
        
        # Validation options
        parser.add_argument(
            '--validate-only',
            action='store_true',
            help='Only validate configuration and input, do not process'
        )
        
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Perform dry run without writing output files'
        )
        
        # Verbosity options
        parser.add_argument(
            '--verbose', '-v',
            action='count',
            default=0,
            help='Increase verbosity (use -v, -vv, or -vvv)'
        )
        
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress non-error output'
        )
        
        return parser
    
    def parse_split_ratios(self, ratios_str: str) -> Dict[str, float]:
        """Parse split ratios from command line string"""
        try:
            ratios = [float(x.strip()) for x in ratios_str.split(',')]
            if len(ratios) != 6:
                raise ValueError("Must provide exactly 6 split ratios")
            
            total = sum(ratios)
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"Split ratios must sum to 1.0, got {total}")
            
            return {
                'd1': ratios[0], 'd2': ratios[1], 'd3': ratios[2],
                'd4': ratios[3], 'd5': ratios[4], 'd6': ratios[5]
            }
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid split ratios format: {e}")
    
    def parse_plugin_list(self, plugins_str: Optional[str]) -> list:
        """Parse plugin list from command line string"""
        if not plugins_str:
            return []
        return [plugin.strip() for plugin in plugins_str.split(',') if plugin.strip()]
    
    def setup_verbosity(self, verbose_count: int, quiet: bool):
        """Setup logging verbosity based on command line options"""
        if quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif verbose_count == 0:
            logging.getLogger().setLevel(logging.INFO)
        elif verbose_count == 1:
            logging.getLogger().setLevel(logging.DEBUG)
        else:  # verbose_count >= 2
            logging.getLogger().setLevel(logging.DEBUG)
            # Enable debug logging for all our modules
            for logger_name in ['app.core', 'app.plugins']:
                logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    def build_configuration_from_args(self, args) -> Dict[str, Any]:
        """Build configuration dictionary from command line arguments"""
        config = {}
        
        # Input/Output configuration
        config['input'] = {
            'file_path': args.input_file,
            'format': 'auto'  # Auto-detect format
        }
        
        config['output'] = {
            'directory': args.output_dir,
            'format': args.output_format,
            'save_metadata': args.save_metadata
        }
        
        # Data processing configuration
        config['data_processing'] = {
            'split_method': args.split_method
        }
        
        if args.split_ratios:
            config['data_processing']['split_ratios'] = self.parse_split_ratios(args.split_ratios)
        
        # Normalization configuration
        config['normalization'] = {
            'method': args.normalization_method,
            'training_datasets': self.parse_plugin_list(args.training_datasets)
        }
        
        # Plugin configuration
        plugins_config = {}
        
        if args.feature_plugins:
            plugins_config['feature_engineering'] = {
                'enabled': self.parse_plugin_list(args.feature_plugins)
            }
        
        if args.postprocessing_plugins:
            plugins_config['postprocessing'] = {
                'enabled': self.parse_plugin_list(args.postprocessing_plugins)
            }
        
        if args.plugin_dirs:
            plugins_config['directories'] = self.parse_plugin_list(args.plugin_dirs)
        
        if plugins_config:
            config['plugins'] = plugins_config
        
        return config
    
    def run(self, args: Optional[list] = None) -> int:
        """Run the CLI interface"""
        try:
            # Parse command line arguments
            parser = self.create_argument_parser()
            parsed_args = parser.parse_args(args)
            
            # Setup logging
            self.setup_verbosity(parsed_args.verbose, parsed_args.quiet)
            
            self.logger.info("Preprocessor System CLI starting...")
            
            # Validate input file
            input_path = Path(parsed_args.input_file)
            if not input_path.exists():
                self.logger.error(f"Input file not found: {input_path}")
                return 1
            
            # Initialize configuration manager
            self.config_manager = ConfigurationManager()
            
            # Load configuration from file if provided
            if parsed_args.config:
                config_path = Path(parsed_args.config)
                if not config_path.exists():
                    self.logger.error(f"Configuration file not found: {config_path}")
                    return 1
                self.config_manager.load_from_file(str(config_path))
            
            # Build configuration from command line arguments
            cli_config = self.build_configuration_from_args(parsed_args)
            self.config_manager.load_from_cli(cli_config)
            
            # Validate configuration
            if not self.config_manager.validate():
                self.logger.error("Configuration validation failed")
                return 1
            
            # Save effective configuration if requested
            if parsed_args.save_config:
                with open(parsed_args.save_config, 'w') as f:
                    json.dump(self.config_manager.merged_config, f, indent=2)
                self.logger.info(f"Configuration saved to: {parsed_args.save_config}")
            
            # If validate-only, stop here
            if parsed_args.validate_only:
                self.logger.info("Configuration validation completed successfully")
                return 0
            
            # Initialize preprocessor core
            self.preprocessor_core = PreprocessorCore(self.config_manager)
            
            # Load input data
            self.logger.info(f"Loading input data from: {input_path}")
            self.preprocessor_core.load_data(str(input_path))
            
            # Execute preprocessing pipeline
            self.logger.info("Executing preprocessing pipeline...")
            result = self.preprocessor_core.process_data()
            
            if not result:
                self.logger.error("Preprocessing pipeline failed")
                return 1
            
            # Export results
            if not parsed_args.dry_run:
                self.logger.info(f"Exporting results to: {parsed_args.output_dir}")
                output_path = Path(parsed_args.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                success = self.preprocessor_core.export_results(
                    str(output_path),
                    format=parsed_args.output_format,
                    include_metadata=parsed_args.save_metadata
                )
                
                if not success:
                    self.logger.error("Failed to export results")
                    return 1
            else:
                self.logger.info("Dry run completed - no files written")
            
            self.logger.info("Preprocessing completed successfully")
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Operation cancelled by user")
            return 130
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            if hasattr(parsed_args, 'verbose') and parsed_args.verbose >= 2:
                import traceback
                traceback.print_exc()
            return 1


def main():
    """Main entry point for CLI"""
    cli = CLIInterface()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())
