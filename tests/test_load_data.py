import pytest

from etl.load_data import data_sources,create_train_test_val

@pytest.fixture(scope="module")
def data():
    return data_sources()

@pytest.fixture(scope="module")
def combined_data(data):
    yield data.merge_order_restaurants()


@pytest.fixture(scope="module")
def final_data(combined_data):
    yield create_train_test_val(combined_data)

def test_data_sources(data):
    assert data.orders.shape[1] == 6
    assert data.restaurants.shape[1] == 4
    assert data.orders.shape[0] == 32394
    assert data.restaurants.shape[0] == 1697


def test_merge_orders_restaurant(combined_data,data):
    assert combined_data.shape[1] == 9
    assert data.orders.shape[0] == combined_data.shape[0]


def test_create_train_test_val(final_data,combined_data):
    final_data = create_train_test_val(combined_data)
    assert final_data.train.shape[0] == 25915
    assert final_data.test.shape[0] == 6479
    assert final_data.test.shape[1] == combined_data.shape[1]