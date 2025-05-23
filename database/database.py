import yaml
from pymongo import MongoClient


def load_database_config():
    try:
        with open("config.yaml") as fp:
            config = yaml.safe_load(fp)
            database_config = config["database"]
            return database_config
    except:
        raise Exception("unable to load database config from config.yaml")


class DatabaseManager:
    def __init__(self, collection):
        self.collection = collection
        config = load_database_config()
        self.name = config["name"]
        self.host = config["host"]
        self.port = config["port"]
        self.db = MongoClient(f"mongodb://{self.host}:{self.port}")[self.name]

    def insert_one(self, item: dict):
        self.db[self.collection].insert_one(item)

    def fetch_collection_as_list(self):
        collection = self.db[self.collection].find()
        col_as_list = list(collection)
        return col_as_list
