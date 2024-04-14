from dataclasses import fields
import logging


def setup_logging():
    # Set up basic configuration for the logging system
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_dataclass_fields(data_cls):
    return {field.name: field.type for field in fields(data_cls)}