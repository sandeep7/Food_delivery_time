import pandas as pd
from scipy.sparse.construct import rand
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from sklearn import metrics
import os
from sklearn import tree
from sklearn.externals import joblib


import argparse

class BaselineModel:
    '''
    class to create the logistic regression
    '''
    def __init__(self,final_data, vars_to_keep , pred_var = 'prep_time_seconds'):
        self.final_data = final_data
        self.vars_to_keep = vars_to_keep
        self.pred_var = pred_var
        self.X_train_scaled , self.X_test_scaled = self.normalize_data()
        self.model = self.create_model()

    def normalize_data(self):
        '''
        Transform and Normalize the inputs so that the feature importance are on the same scale
        '''
        print("normalize the data")
        df_train = pd.get_dummies(self.final_data.train[self.vars_to_keep] ,drop_first=True)
        print("train dummies shape",df_train.shape[1])
        df_test = pd.get_dummies(self.final_data.test[self.vars_to_keep] ,drop_first=True)
        print("test dummies shape",df_test.shape[1])
        ss = MinMaxScaler()
        # X_train_scaled = ss.fit_transform(df_train)
        # X_train_scaled = pd.DataFrame(X_train_scaled,columns=df_train.columns)
        # X_test_scaled = ss.transform(df_test)
        # X_test_scaled = pd.DataFrame(X_test_scaled,columns=df_test.columns)
        X_train_scaled = df_train.copy()
        X_test_scaled = df_test.copy()
        return X_train_scaled , X_test_scaled
    

    def create_model(self):
        print("create the linear model using ridge with alpha 2")

        model = Ridge(random_state=999 , alpha = 2)
        
        model.fit(self.X_train_scaled,self.final_data.train[self.pred_var])
        return model


    def print_evaluation_statistics(self,train = True):
        if train:
            pred = self.model.predict(self.X_train_scaled)
            print("Mean Absolute Error:", metrics.mean_absolute_error(self.final_data.train['prep_time_seconds'],pred))
            print("Mean Square Error :",metrics.mean_squared_error(self.final_data.train['prep_time_seconds'],pred))
        else:
            pred = self.model.predict(self.X_test_scaled)
            print("Mean Absolute Error :",metrics.mean_absolute_error(self.final_data.test['prep_time_seconds'],pred))
            print("Mean Square Error :",metrics.mean_squared_error(self.final_data.test['prep_time_seconds'],pred))
            
class create_train_test_val:
    '''
    Class to create the train test and val
    '''
    def __init__(self,
                 data):
        self.data = data
        self.train, self.test = self.create_test_train()
    
    def create_test_train(self):

        train, test = train_test_split(self.data, random_state = 999,test_size=0.2)

        print(len(train), 'train examples')
        print(len(test), 'test examples')

        return train,test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    #Saves Checkpoints and graphs
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    #Save model artifacts
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    #Train data
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
 
    args = parser.parse_args()

    file = os.path.join(args.train, "combined_data.csv")
    final_data = pd.read_csv(file, engine="python")
    
    final_data = create_train_test_val(final_data)
    
    # Creating the dummy variables 
    base_model = BaselineModel(final_data,vars_to_keep=['order_value_gbp',
                                              'number_of_items',
                                              'country',
                                              'city',
                                              'type_of_food',
                                              'value_per_item',
                                              'hour_order_acknowledged_at',
                                              'day_of_order'])
    clf = base_model.create_model()
    joblib.dump(clf, os.path.join(args.model_dir, "lr_model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    regressor = joblib.load(os.path.join(model_dir, "lr_model.joblib"))
    return regressor