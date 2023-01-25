# Extracts data from json file and returns a list of dictionaries
import json


class JsonExtractor:
    """
    Example usage
    json_file = 'data.json'
    json_extractor = JsonExtractor(json_file)
    data = json_extractor.extract_json_data()
    """

    def __init__(self, json_file):
        self.json_file = json_file

    def extract_json_data(self):
        with open(self.json_file) as f:
            data = json.load(f)
        return data
