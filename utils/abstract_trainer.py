from abc import ABC, abstractmethod
from database.database import DatabaseManager
import time
from uuid import uuid4
import dill 
import base64


class RLTrainer(ABC):
    def __init__(self, collection: str):
        self.collection = collection
        self.db = DatabaseManager(collection)
        self.rewards = []
        self.num_epoch = 0
        self.start_time = time.time()
        self.trainer_id = str(uuid4())

    @abstractmethod
    def epoch(self):
        """
        Perform the whole epoch, return the obtained reward
        """
        pass

    def upload_trainer_to_trainers_collection(self):
        """
        Compress the full class into dill format and uploads it into a
        trainers collection in b64 format
        """
        trainers_db = DatabaseManager("trainers")
        bytes_obj = dill.dumps(self)
        b64_obj = base64.b64encode(bytes_obj).decode()
        print(b64_obj)
        total_time = time.time() - self.start_time
        data = {
                "trainer_id":self.trainer_id,
                "collection": self.collection,
                "epochs": self.num_epoch,
                "total_time": total_time,
                "trainer_b64": b64_obj,
                }
        print(f"trainer {self.trainer_id} uploaded to trainers collection")
        trainers_db.insert_one(data)
        return data

    def finish(self):
        final_t = time.time()
        total_time = final_t - self.start_time
        print(f"trainer {self.trainer_id} took {total_time/3600} hours for {self.num_epoch} epochs") 
        return total_time

    def train(self, epochs):
        print(f"started training with trainer: {self.trainer_id}")
        for i in range(epochs):
            self.num_epoch += 1
            start = time.time()
            reward = self.epoch()
            end = time.time()
            total_time = end - start
            data = {
                    "trainer_id": self.trainer_id,
                    "epoch": i,
                    "reward": reward,
                    "duration": total_time
                    }
            self.db.insert_one(data)
        # self.upload_trainer_to_trainers_collection() # It's impossible 
        self.finish()
        return data
