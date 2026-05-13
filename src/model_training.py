import os
from src.logger import getlogger
from src.custom_exeption import CustomExeption
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt


logger = getlogger(__name__)

class ModelTraining:
    def __init__(self):
        self.processed_data_path = "artifacts/processed"
        self.modle_path = "artifacts/models"
        os.makedirs(self.modle_path,exist_ok=True)
        self.model = DecisionTreeClassifier(criterion='gini',max_depth=30,random_state=42)
        logger.info("Model trainig initialized..")

    def load_data(self):
        try:
            x_train = joblib.load(os.path.join(self.processed_data_path,"x_train.pkl"))
            x_test = joblib.load(os.path.join(self.processed_data_path,"x_test.pkl"))
            y_train = joblib.load(os.path.join(self.processed_data_path,"y_train.pkl"))
            y_test = joblib.load(os.path.join(self.processed_data_path,"y_test.pkl"))

            logger.info("Data loaded succesfully...")
            return x_train,x_test,y_train,y_test


        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomExeption("Error while loading data",e)
        
    def train_model(self,x_train,y_train):
        try:
            self.model.fit(x_train,y_train)
            joblib.dump(self.model, os.path.join(self.modle_path,"model.pkl"))

            logger.info("Model train and saved successfully..")
        
        except Exception as e:
            logger.error(f"Error while model training {e}")
            raise CustomExeption("Error while modle training",e)
        

    def evaluate_model(self,x_test,y_test):
        try:
            y_pred = self.model.predict(x_test)
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred,average="weighted")
            f1 = f1_score(y_test,y_pred,average="weighted")

            logger.info(f"Accuracy_score: {accuracy}")
            logger.info(f"Precision_score: {precision}")
            logger.info(f"F1_score: {f1}")

            cm = confusion_matrix(y_test,y_pred)
            plt.figure(figsize=(8,5))
            sns.heatmap(cm,annot=True,cmap='Blues',xticklabels=np.unique(y_test),yticklabels=np.unique(y_test))
            plt.title("Confusion matrix")
            plt.xlabel("Predicted lable")
            plt.ylabel("Actual lable")
            confusion_matrix_path = os.path.join(self.modle_path,"confusion_matrics.png")
            plt.savefig(confusion_matrix_path)
            plt.close()

            logger.info("Confusion matrix save successfully...")

        except Exception as e:
            logger.error(f"Error while evaluating..{e}")
            raise CustomExeption("Error while evaluating",e)
        
    def run(self):
        x_train,x_test,y_train,y_test = self.load_data()
        self.train_model(x_train,y_train)
        self.evaluate_model(x_test,y_test)

if __name__=="__main__":
    trainer = ModelTraining()
    trainer.run()