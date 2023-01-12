from re import L
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers

from etl.feature_generation import GenerateFeatures


class NNmodel:
    def __init__(self,final_data,combined_data,feature_names,outcome_name = 'prep_time_seconds'):
        self.final_data = final_data
        self.combined_data = combined_data
        self.feature_names = feature_names
        self.outcome_name = outcome_name
        self.train_ds ,self.val_ds , self.test_ds = self.prepare_data()
        self.features = self.prepare_feature_layer()
        self.model = self.model_building()



    def prepare_data(self):
        '''
        Create TF datasets and add median features
        return: (data, train_ds, val_ds, test_ds) (pandas_df, tf.Dataset, tf.Dataset, tf.Dataset)
        The original dataset and train, validation and test tensorflow datasets
        '''

        train , val  = train_test_split(self.final_data.train,random_state= 999, test_size=0.2)


        batch_size = 32
        train_ds = self.df_to_dataset(train, features=self.feature_names, outcome=self.outcome_name, batch_size=batch_size)
        val_ds =  self.df_to_dataset(val, shuffle=False, features=self.feature_names, outcome=self.outcome_name, batch_size=batch_size)
        test_ds =  self.df_to_dataset(self.final_data.test, features=self.feature_names, outcome=self.outcome_name, shuffle=False, batch_size=batch_size)

        return train_ds , val_ds , test_ds


    # A utility function to create a tf.data dataset from a Pandas Dataframe
    @staticmethod
    def df_to_dataset(dataframe, features, outcome, shuffle=True, batch_size=128):
        dataframe = dataframe.copy()
        dataframe[outcome] = dataframe[outcome].astype(float)
        y = dataframe.pop(outcome)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe[features]), y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    @staticmethod
    def get_normalization_parameters(traindf, features):
        def _z_score_params(column):
            mean = traindf[column].mean()
            std = traindf[column].std()
            return {'mean': mean, 'std': std}

        normalization_parameters = {}
        for column in features:
             normalization_parameters[column] = _z_score_params(column)
        return normalization_parameters


    def prepare_feature_layer(self):
        """
        given the dataset, prepares the input to the tensorflow model
        :param data: pandas dataset to be used in the modeling
        :return: a tensorflow Features Layer object to serve as model layer
        """

        print("create Features to feed in the DL model")

        NUMERIC_FEATURES = ['rest_median_order_value_gbp', 'rest_median_number_of_items',
                    'rest_median_value_per_item', 'number_of_items', 'order_value_gbp','value_per_item']

       
        normalization_parameters = self.get_normalization_parameters(self.final_data.train,
                                                            NUMERIC_FEATURES)
        feature_columns = []

        # numeric cols
        for header in NUMERIC_FEATURES:
            column_params = normalization_parameters[header]
            mean = column_params['mean']
            std = column_params['std']
            def normalize_column(col): 
                return (col - mean)/std
            normalizer_fn = normalize_column
            feature_columns.append(feature_column.numeric_column(key = header,dtype=tf.dtypes.float32,normalizer_fn=normalizer_fn))

        # indicator cols - country
        country = feature_column.categorical_column_with_vocabulary_list(
            'country', list(self.combined_data['country'].unique()))
        country_one_hot = feature_column.indicator_column(country)
        feature_columns.append(country_one_hot)

        # indicator cols - day_of_order
        day_of_order = feature_column.categorical_column_with_vocabulary_list(
            'day_of_order', list(self.combined_data['day_of_order'].unique()))
        day_of_order_one_hot = feature_column.indicator_column(day_of_order)
        feature_columns.append(day_of_order_one_hot)

        # indicator cols - hour of acknowledgement
        hour_order_acknowledged_at = feature_column.categorical_column_with_vocabulary_list(
            'hour_order_acknowledged_at', list(self.combined_data['hour_order_acknowledged_at'].unique()))
        hour_order_acknowledged_at_one_hot = feature_column.indicator_column(hour_order_acknowledged_at)
        feature_columns.append(hour_order_acknowledged_at_one_hot)

        # city embedding
        city = feature_column.categorical_column_with_vocabulary_list(
            'city', list(self.combined_data['city'].unique()))
        city_embedding = feature_column.embedding_column(city, dimension=4)
        feature_columns.append(city_embedding)

        # type_of_food embedding
        type_of_food = feature_column.categorical_column_with_vocabulary_list(
            'type_of_food', list(self.combined_data['type_of_food'].unique()))
        type_of_food_embedding = feature_column.embedding_column(type_of_food, dimension=8)
        feature_columns.append(type_of_food_embedding)

        # restaurant_id embedding
        restaurant_id = feature_column.categorical_column_with_vocabulary_list(
            'restaurant_id', list(self.combined_data['restaurant_id'].unique()))
        restaurant_id_embedding = feature_column.embedding_column(restaurant_id, dimension=22)
        feature_columns.append(restaurant_id_embedding)

        # crossed cols
        crossed_feature = feature_column.crossed_column([hour_order_acknowledged_at, day_of_order], hash_bucket_size=100)
        crossed_feature = feature_column.indicator_column(crossed_feature)
        feature_columns.append(crossed_feature)

        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

        return feature_layer


    def model_building(self):
        """
        builds and returns tensorflow model
        : input param feature_layer: the feature layer built from the dataset
        :return: tensforflow model
        """

        model = tf.keras.Sequential([
            self.features,
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)])

        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(optimizer=optimizer,
                    loss='mse',
                    metrics=['mae', 'mse'])
        return model
    

    def model_fit_and_evaluate(self,epochs):
        """
        fits the model and presents the model error
        :param model: tensorflow model to train
        :param train_ds: tf.Dataset of training samples
        :param val_ds: tf.Dataset of validation samples
        :param test_ds: tf.Dataset of test samples
        :param epochs: the number of epochs to run through the data
        :return: None - print errors to standard out
        """

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10)
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                            filepath="model_weights/weights.cpkt",
                                            monitor='val_loss',
                                            save_best_only=True)
        self.model.fit(self.train_ds,
                verbose=2,
                validation_data=self.val_ds,
                callbacks=[early_stop,model_checkpoint_callback],
                epochs=epochs)

        los, mae, mse = self.model.evaluate(self.test_ds, verbose=1)
        print('test_loss', los)
        print('test mae', mae)
        print('test mse', mse)