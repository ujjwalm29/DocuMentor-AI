from dataclasses import fields
import logging
import time

logger = logging.getLogger(__name__)

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        class_name = args[0].__class__.__name__ if args and hasattr(args[0], '__class__') else 'N/A'
        logger.info(f"{class_name}.{func.__name__} executed in {end_time - start_time} seconds")
        return result
    return wrapper


def async_time_function(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)  # Await the async function
        end_time = time.time()
        class_name = args[0].__class__.__name__ if args and hasattr(args[0], '__class__') else 'N/A'
        logger.info(f"{class_name}.{func.__name__} executed in {end_time - start_time} seconds")
        return result
    return wrapper


def setup_logging():
    # Set up basic configuration for the logging system
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_dataclass_fields(data_cls):
    return {field.name: field.type for field in fields(data_cls)}