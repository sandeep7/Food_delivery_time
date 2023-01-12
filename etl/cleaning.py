import pandas as pd
from collections import Counter

class RemoveOutlierData:
    '''
    This class is used to clean the data and remove the skewness
    '''

    @staticmethod
    def filter_orders_data(orders,max_prep_seconds=3600): 
        '''
        filter the orders table based on prep time to remove the skewness in the data
        input param : values from which to remove the data points
        return param : filtered orders dataframe
        '''

        orders = orders.loc[orders['prep_time_seconds'] < max_prep_seconds]

        print('size of orders after removals', orders.shape[0])

        return orders

    
    @staticmethod
    def cap_bin_orders_data(orders, max_items=25, max_value=200): 
        '''
        Cap and bin the orders table based on items and values to remove the skewness in the data
        input param : values from which to remove the data points
        return param : filtered orders dataframe
        '''
        # orders with more than 25 items (confer number_of_items_hist.html)
        orders['number_of_items'] = orders['number_of_items'].clip(upper = max_items)

        # orders that total more than 200 in local currency (GBP/EUR) (confer order_value_hist.html)
        orders['order_value'] = orders['order_value_gbp'].clip(upper = max_value)

        return orders

    
    @staticmethod
    def filter_on_location_and_food_type(combined_data):
        """
        r emoves outliers for the model
        input param: pandas dataframe of orders
        return param :  pandas dataframe of orders and restaurants after filtering
        """

        # keep only cities and food types with more than 100 orders
        city_counter = Counter(combined_data['city'])
        cities_to_keep = [c for c, n in city_counter.items() if n > 100]

        food_type_counter = Counter(combined_data['type_of_food'])
        food_type_to_keep = [c for c, n in food_type_counter.items() if n > 100]

        order_and_rest_filtered = combined_data.loc[combined_data['city'].isin(cities_to_keep)]
        order_and_rest_filtered = order_and_rest_filtered.loc[order_and_rest_filtered['type_of_food'].isin(food_type_to_keep)]

        print('final size of dataset after filtering', order_and_rest_filtered.shape)

        return order_and_rest_filtered