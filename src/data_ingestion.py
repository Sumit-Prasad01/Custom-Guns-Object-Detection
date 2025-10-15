import os
import shutil
import zipfile
import kagglehub
from src.logger import get_logger
from config.data_ingestion_config import *
from src.custom_exception import CustomException

logger = get_logger(__name__)

class DataIngestion:

    def __init__(self, dataset_name : str, target_dir : str):

        self.dataset_name = dataset_name
        self.target_dir = target_dir

    def create_raw_dir(self):

        raw_dir = os.path.join(self.target_dir, "raw")
        if not os.path.exists(raw_dir):
            try:
                os.makedirs(raw_dir)
                logger.info(f"Created the {raw_dir}")

            except Exception as e:
                logger.error(f"Error while creating {raw_dir}, {e}")
                raise CustomException(f"Failed to create {raw_dir}, {e}")
            
        return raw_dir

    def extract_iamges_and_labels(self, path : str, raw_dir : str):

        try:
            if path.endswith('.zip'):
                logger.info("Extracting Zip File.")

                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(path)

                
            images_folder = os.path.join(path, "Images")
            labels_folder = os.path.join(path, "Labels")

            # Images Folder 
            if os.path.exists(images_folder):
                shutil.move(images_folder, os.path.join(raw_dir,"Images"))

                logger.info("Images moved successfully.")

            else:
                logger.info("Iamges folder don't exist.")

            # Labels Folder
            if os.path.exists(labels_folder):
                shutil.move(labels_folder, os.path.join(raw_dir,"Labels"))

                logger.info("Labels moved successfully.")

            else:
                logger.info("Labels folder don't exist.", e)


        except Exception as e:
                logger.error(f"Error while Extracting {images_folder}, {labels_folder}")
                raise CustomException(f"Failed to Extract {images_folder}, {labels_folder}")
        

    def download_dataset(self, raw_dir : str):

        try:
            path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Downloaded the data successfully from {path}")

            self.extract_iamges_and_labels(path, raw_dir)
        
        except Exception as e:
                logger.error(f"Error while downloading dataset from {self.dataset_name}, {path}, {e}")
                raise CustomException(f"Failed to download dataset from {self.dataset_name}, {path}, {e}")
        
    
    def run(self):

        try:
            raw_dir = self.create_raw_dir()
            self.download_dataset(raw_dir)
        
        except Exception as e:
                logger.error(f"Error while data ingestion pipeline {e}")
                raise CustomException(f"Failed to run data ingesyion pipeline {e}")
        

if __name__ == "__main__":
     
    ingest = DataIngestion(DATASET_NAME, TARGET_DIR)
    ingest.run()