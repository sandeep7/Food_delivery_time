import pytest


from etl.feature_generation import GenerateFeatures
from etl.load_data import data_sources,create_train_test_val

@pytest.fixture(scope="module")
def data():
    return data_sources()

@pytest.fixture(scope="module")
def combined_data(data):
    yield data.merge_order_restaurants()



# combined_data = data.merge_order_restaurants()  
# final_data = create_train_test_val(combined_data)


def test_generate_order_features(combined_data,data):

    combined_data = GenerateFeatures.generate_order_features(combined_data)
    

    assert "value_per_item" in combined_data.columns
    assert "hour_order_acknowledged_at" in combined_data.columns
    assert "day_of_order" in combined_data.columns
    assert data.orders.shape[0] == combined_data.shape[0]


def test_generate_order_features_median(combined_data,data):


    final = create_train_test_val(combined_data)
    final = GenerateFeatures.restaurant_feature_generation_from_orders(final)

    assert "rest_median_order_value_gbp" in final.train.columns
    assert "rest_median_number_of_items" in final.train.columns
    assert "rest_median_value_per_item" in final.train.columns
    # The Actual number of rows liable to change as cleanup not performed in this test. need to create mock data to better perform tests