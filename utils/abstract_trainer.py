from abc import ABC, abstractmethod
from database.database import DatabaseManager
import time
from uuid import uuid4


class RLTrainer(ABC):
    def __init__(self, collection: str, epochs = 100_000):
        self.collection = collection
        self.db = DatabaseManager(collection)
        self.rewards = []
        self.epochs = epochs
        self.start_time = time.time()
        self.trainer_id = str(uuid4())


    @abstractmethod
    def epoch(self):
        """
        Perform the whole epoch, return the reward
        """
        pass

    def finish(self):
        final_t = time.time()
        total_time = final_t - self.start_time
        print(f"whole process took {total_time/3600} hours for {self.epoch} epochs") 
        return total_time

    def upload_trainer_to_trainers_collection(self):
        """
        Compress the full class into pickle format and uploads it into a
        trainers collection in b64 format
        """
        trainers_db = DatabaseManager("trainers")
        data = {
                "trainer_id":self.trainer_id,
                "collection": self.collection,
                "trainer_b64": ""# pending to fill it
                }
        trainers_db.insert_one(data)
        print(f"trainer {self.trainer_id} uploaded to trainers collection")

    def train(self):
        print(f"started training with trainer: {self.trainer_id}")
        for i in range(self.epochs):
            start = time.time()
            reward = self.epoch()
            end = time.time()
            total_time = end - start
            data = {
                    "trainer_id": self.trainer_id
                    "epoch": i,
                    "reward": reward,
                    "duration": total_time
                    }
            self.db.insert_one(data)
        self.finish()
        self.upload_trainer_to_trainers_collection()
