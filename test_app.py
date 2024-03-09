import pytest
from streamlit.testing.v1 import AppTest
from app_git.app import get_customers_ids, get_customer_values, get_features_selected



@pytest.fixture
def streamlit_client():
    script_path = "app_git/app.py"

    # Initialise l'application
    app_test = AppTest.from_file(script_path, default_timeout=50)
    # lance l'appli
    app_test.run()

    yield app_test

def test_get_customers_ids(streamlit_client):
    customers_ids = get_customers_ids()
    assert isinstance(customers_ids, list)


def test_get_customer_values(streamlit_client):
    customer_id = 100002
    customer_values = get_customer_values(customer_id)
    assert isinstance(customer_values, dict)

def test_get_features_selected(streamlit_client):
    features_selected_list = get_features_selected()
    assert isinstance(features_selected_list, list)


if __name__ == "__main__":
    pytest.main(["-v", "--capture=no", "test_app.py"])