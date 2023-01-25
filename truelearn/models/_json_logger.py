import logging
from truelearn.models._json_extractor import JsonExtractor


# Logs the data in JSON format
class JsonLogger:
    """
    Example usage
    json_logger = JsonLogger('data.json')
    json_logger.log_data()
    """

    def __init__(self, json_file):
        self.json_file = json_file

    def log_data(self):
        json_extractor = JsonExtractor(self.json_file)
        data = json_extractor.extract_json_data()
        for item in data:
            logging.info(item)
