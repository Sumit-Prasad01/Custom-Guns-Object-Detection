import os
import torch 
from torch import optim
from torch.utils.data import random_split, DataLoader
from src.model_architecture import FasterRCNNModel
from src.data_processing import GunDataset
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from torch.utils.tensorboard import SummaryWriter
import time


logger = get_logger(__name__)

model_save_path = "artifacts/models/"
os.makedirs(model_save_path, exist_ok = True)


class ModelTraining:

    def __init__(self, model_class, num_classes, learning_rate , epochs, dataset_path, device):

        self.model_class = model_class
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.device = device

        # Tensorboard

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"tensorboard_logs/{timestamp}"
        os.makedirs(self.log_dir, exist_ok = True)

        self.writer = SummaryWriter(log_dir = self.log_dir)
       
        try:
            self.model = self.model_class(self.num_classes, self.device).model
            self.model.to(self.device)
            logger.info("Model moved to device")

            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)

            logger.info("Optimizer has been initialized.")
        
        except Exception as e:
            logger.info(f"Failed to initialize  model training {e}")
            raise CustomException(f"Failed to initialize  model training", e)
        


    def collate_fn(self, batch):
        try:
            return tuple(zip(*batch))
        
        except Exception as e:
            logger.info(f"Failed to batch data {e}")
            raise CustomException(f"Failed to batch data", e)



    def split_data(self):
        try:
            dataset = GunDataset(self.dataset_path, self.device)
            dataset = torch.utils.data.Subset(dataset, range(5))

            train_size = int(0.8*len(dataset))
            val_size = len(dataset) - train_size

            train_dataset , val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size = 3, shuffle = True, num_workers = 0, collate_fn = self.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size = 3, shuffle = False, num_workers = 0, collate_fn = self.collate_fn)

            logger.info("Data Splitted Successfully.")

            return train_loader, val_loader
        
        except Exception as e:
            logger.info(f"Failed to split data {e}")
            raise CustomException(f"Failed to split data", e)
        


    def train(self):
        try:
            train_loader, val_loader = self.split_data()

            for epoch in range(self.epochs):
                logger.info(f"Starting epoch {epoch}")
                self.model.train()

                for i, (images, targets) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    losses = self.model(images, targets)

                    if isinstance(losses, dict):
                        total_loss = 0

                        for key, value in losses.items():
                            if isinstance(value, torch.Tensor):
                                total_loss += value
                            
                        if total_loss == 0:
                            logger.error("There was error in loss capturing")

                            raise ValueError("Total Value is ZERO....")
                        
                        self.writer.add_scalar("Loss/train", total_loss.item(), epoch*len(train_loader) + i)
                        
                    else:
                        total_loss = losses[0]

                        self.writer.add_scalar("Loss/train", total_loss.items(), epoch*len(train_loader) + i)

                    total_loss.backward()
                    self.optimizer.step()

                self.writer.flush()

                self.model.eval()

                with torch.no_grad():
                    for images, targets in val_loader:
                        val_losses = self.model(images, targets)

                        logger.info(type(val_losses))
                        logger.info(f"VAL LOSS : {val_losses}")
                
                model_path = os.path.join(model_save_path, "fasterrcnn.pth")
                torch.save(self.model.state_dict(), model_path)

                logger.info("Model Saved Successfully.")

        except Exception as e:
            logger.info(f"Failed to train model {e}")
            raise CustomException(f"Failed to train model ", e)
        

if __name__ == "__main__":
    try:
        
        training = ModelTraining(
                                model_class = FasterRCNNModel,
                                num_classes = 2,
                                learning_rate = 0.0001,
                                epochs = 1,
                                dataset_path = ROOT_PATH,
                                device = DEVICE
                            )
        
        training.train()
    
    except Exception as e:
            logger.info(f"Failed to run training pipeline {e}")
            raise CustomException(f"Failed to run training pipeline ",e)