from condenses_scoring.schemas import Message, ScoringRequest
import pytest
import requests
import os


@pytest.fixture
def base_url():
    host = os.getenv("CONDENSES_SCORING_HOST", "localhost")
    port = os.getenv("CONDENSES_SCORING_PORT", "8000")
    return f"http://{host}:{port}"


def test_scoring_endpoint(base_url):
    # Prepare test data
    test_request = ScoringRequest(
        original_messages=[
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing well, thank you!"),
        ],
        compressed_messages=[
            Message(role="user", content="Hi"),
            Message(role="assistant", content="I'm doing well, thank you!"),
        ],
    )

    # Make request to the endpoint
    response = requests.post(f"{base_url}/api/scoring", json=test_request.model_dump())

    # Assert response
    assert response.status_code == 200
    assert "score" in response.json()
    assert isinstance(response.json()["score"], float)
    assert 0 <= response.json()["score"] <= 1


def test_scoring_endpoint_empty_messages(base_url):
    # Test with empty messages
    test_request = ScoringRequest(original_messages=[], compressed_messages=[])

    response = requests.post(f"{base_url}/api/scoring", json=test_request.model_dump())
    assert response.status_code == 422  # FastAPI validation error


def test_scoring_endpoint_invalid_request(base_url):
    # Test with invalid request format
    response = requests.post(f"{base_url}/api/scoring", json={"invalid": "data"})
    assert response.status_code == 422
