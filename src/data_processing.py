from src.custom_exeption import CustomExeption
from src.logger import getlogger
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import os

logger = getlogger(__name__)

class DataProcessing:
    def __init__(self,file_path):
        self.file_path = file_path
        self.df = None
        self.processed_data_path = "artifacts/processed"
        os.makedirs(self.processed_data_path,exist_ok=True) 

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info("Data loaded successfully...")
        except Exception as e:
            logger.error(f"Error while data loading... {e}")
            raise CustomExeption("Error while data loading.",e)
        
    def handle_outliers(self,column:str):
        try:
            logger.info("Handling outliers...")
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)

            IQR = Q3-Q1

            lower_value = Q1 - IQR*1.5
            upper_value = Q3 + IQR*1.5
            sepel_median = np.median(self.df[column])

            for i in self.df["SepalWidthCm"]:
                if i> upper_value or i < lower_value:
                    self.df["SepalWidthCm"] = self.df[column].replace(i,sepel_median)
            
            logger.info("Handle outliers")
        
        except Exception as e:
            logger.error(f"Error hapening while handling outliers {e}")
            raise CustomExeption("Error hapening while handling outliers",e)
        
    def train_test_split(self):
        try:
            x = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
            y = self.df[["Species"]]

            x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42)
            logger.info("Data splitted")

            joblib.dump(x_train,os.path.join(self.processed_data_path,"x_train.pkl"))
            joblib.dump(x_test,os.path.join(self.processed_data_path,"x_test.pkl"))
            joblib.dump(y_train,os.path.join(self.processed_data_path,"y_train.pkl"))
            joblib.dump(y_test,os.path.join(self.processed_data_path,"y_test.pkl"))

            logger.info("Saved datafiles...")


        except Exception as e:
            logger.error("Error hapening while spliting... {e}")
            raise CustomExeption("Error hapening while spliting data",e)
        
    
    def run(self):
        self.load_data()
        self.handle_outliers("SepalWidthCm")
        self.train_test_split()

if __name__=="__main__":
    processor = DataProcessing("artifacts/raw/data.csv")
    processor.run()