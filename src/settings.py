import json

class Settings:
    def __init__(self, path) -> None:
        self.path= path
        with open(path) as settings:
            self.data = json.load(settings)

    def get(self, setting_name, default=None):
        return self.data.get(setting_name, default)["value"]

    def describe(self, setting_name):
        return self.data.get(setting_name, None)["description"]

    def get_type(self, setting_name):
        return self.data.get(setting_name, None)["type"]

    def __str__(self) -> str:
        return str(self.data)