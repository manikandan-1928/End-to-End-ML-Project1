import os
from src.mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up
        
    def scaling_data(self):
        data = pd.read_csv(self.config.data_path)

        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(data.drop('quality', axis=1))
        scaled_df = pd.DataFrame(scaled_data, columns=data.drop('quality', axis=1).columns)
                                                                  
        logger.info("Transformed into standard scaled data")

        return scaled_df


    def train_test_spliting(self, scaled_df):
        data = scaled_df

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)