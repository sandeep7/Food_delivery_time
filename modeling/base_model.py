import pandas as pd
from scipy.sparse.construct import rand
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import metrics
import plotly.graph_objs as go
import plotly as py
import plotly.offline as pyo
from plotly.subplots import make_subplots

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
        self.X_train_scaled.to_csv('test_1.csv')
        return model.fit(self.X_train_scaled,self.final_data.train[self.pred_var])

    
    def plot_variable_importance(self):

        importances = pd.DataFrame(data={
                                'Attribute': self.X_train_scaled.columns,
                                'Importance': self.model.coef_})
        
        importances = importances.sort_values(by='Importance', ascending=False)
        # importances['Importance'] = importances['Importance'].clip(lower = -100 , upper= 100)
        

        fig = make_subplots(rows=1,
                            cols=1)


        fig.add_trace(
            go.Bar(x=importances['Attribute'],
                    y=importances['Importance'],
                    name = "Feature Importance"), row=1, col=1)

        # Update xaxis properties
        fig.update_xaxes(title_text="Feature Name", row=1, col=1)
        # Update yaxis properties
        fig.update_yaxes(title_text="%", row=1, col=1)


        fig.update_layout(height=500,bargap=0.2)


        fig.show()

        


    def print_evaluation_statistics(self,train = True):
        if train:
            pred = self.model.predict(self.X_train_scaled)
            print("Mean Absolute Error :",metrics.mean_absolute_error(self.final_data.train['prep_time_seconds'],pred))
            print("Mean Square Error :",metrics.mean_squared_error(self.final_data.train['prep_time_seconds'],pred))
        else:
            pred = self.model.predict(self.X_test_scaled)
            print("Mean Absolute Error :",metrics.mean_absolute_error(self.final_data.test['prep_time_seconds'],pred))
            print("Mean Square Error :",metrics.mean_squared_error(self.final_data.test['prep_time_seconds'],pred))