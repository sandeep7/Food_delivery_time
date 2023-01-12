import pandas as pd
from sklearn.model_selection import train_test_split

class data_sources:
    '''
    Class to load the data and correct the time stamps
    parameters:
    restaurants_url : location of the restaurant file in csv.gz format
    orders_url : Location of the orders file in csv.gz format
    '''
    def __init__(self,
                 restaurants_url = 'raw_data/restaurants.csv.gz',
                 orders_url = 'raw_data/orders.csv.gz'):
        self.orders = self.read_orders_data(orders_url)
        self.restaurants = self.read_restaurants_data(restaurants_url)
    
    def read_orders_data(self,orders_url):
        df = pd.read_csv(orders_url)
#       process and convert the hours and as mentioned put UTC is true
        df['order_acknowledged_at'] = pd.to_datetime(df['order_acknowledged_at'], utc=True)
        df['order_ready_at'] = pd.to_datetime(df['order_ready_at'], utc=True)        
        return df
    
    def read_restaurants_data(self,restaurants_url):
        return pd.read_csv(restaurants_url)
    
    def merge_order_restaurants(self):
        return self.orders.merge(right=self.restaurants, how='left', on='restaurant_id')



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