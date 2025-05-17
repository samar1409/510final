# tests/test_app.py
import pytest
from app import app as flask_app # Import your Flask app instance

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # You might add test-specific configurations here later
    # app.config.update({"TESTING": True, ...})
    yield flask_app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

def test_home_page_loads(client):
    """Test if the home page loads successfully (HTTP 200)."""
    response = client.get('/')
    assert response.status_code == 200
    print(f"Response data sample: {response.data[:200]}") # Optional: print response start
    # You can also test for specific content
    assert b"King County Real Estate Dashboard" in response.data # Check if title is in HTML
    assert b'<div id="map">' in response.data # Check if map div exists

