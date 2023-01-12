import pandas as pd
from collections import Counter

class GenerateFeatures:


    @staticmethod
    def generate_order_features(data):
        """
        generates new features for the orders dataset and returns it
        input params: pandas dataframe of the orders dataset
        return params: pandas dataframe with more features including the outcome to model
        """

        # average price per item for the restaurant (a proxy for the type of restaurant)
        data['value_per_item'] = data['order_value_gbp'] / data['number_of_items']

        # hour of the day for the order (kitchens will have busier and less busy times)
        data['hour_order_acknowledged_at'] = data['order_acknowledged_at'].dt.hour

        # day of the week for the order (kitchens will have busier and less busy times)
        data['day_of_order'] = data['order_acknowledged_at'].dt.dayofweek #  Monday=0, Sunday=6.

        # Convert to string as they need to be treated as categorical variable
        data['day_of_order']  = data['day_of_order'].astype(str)
        data['hour_order_acknowledged_at']  = data['hour_order_acknowledged_at'].astype(str)

        return data
    
    @staticmethod
    def restaurant_feature_generation_from_orders(final_data):
        """
        generate features at the restaurant level
        input params: pandas dataframe of the orders dataset
        return params: pandas dataframe with more features including the outcome to model
        """
        print('shape of original train data', final_data.train.shape[0])
        print('shape of original test data', final_data.test.shape[0])
        # some medians at the restaurant level
        orders_grouped_median = final_data.train.groupby(by='restaurant_id').median()[['order_value_gbp', 'number_of_items', 'value_per_item']]

        orders_grouped_median.rename(mapper={'order_value_gbp': 'rest_median_order_value_gbp',
                                            'number_of_items': 'rest_median_number_of_items',
                                            'value_per_item': 'rest_median_value_per_item'},
                                    axis ='columns',
                                    inplace=True)

        orders_grouped_median.reset_index(level=0, inplace=True)

        final_data.train = final_data.train.merge(orders_grouped_median, how='inner', on='restaurant_id')
        final_data.test = final_data.test.merge(orders_grouped_median, how='inner', on='restaurant_id')

        print('shape of train data after transformation', final_data.train.shape[0])
        print('shape of test data after transformation', final_data.test.shape[0])

        return final_data