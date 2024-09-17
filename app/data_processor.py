from app.data_handler import load_csv, write_csv

def run_preprocessor_pipeline(config, plugin):
    """Process the data using the specified plugin."""
    data = load_csv(config['input_file'])

    # Debugging: Print loaded data
    print("Loaded data:\n", data.head())

    processed_data = plugin.process(data)

    # Debugging: Print processed data
    print("Processed data:\n", processed_data.head())

    include_date = config['force_date']  if 'date' in processed_data.columns else False

    if not config['quiet_mode']:
        print("Processing complete. Writing output...")

    write_csv(config['output_file'], processed_data, include_date=include_date, headers=config['headers'])

    if not config['quiet_mode']:
        print(f"Output written to {config['output_file']}")
    return processed_data
