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
    # Prepare test data with valid alternating user/assistant messages
    test_request = ScoringRequest(
        original_messages=[
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing well, thank you!"),
            Message(role="user", content="That's great!"),
            Message(role="assistant", content="Indeed it is!"),
        ],
        compressed_messages=[
            Message(role="user", content="Hi"),
            Message(role="assistant", content="I'm doing well, thank you!"),
            Message(role="user", content="Great!"),
            Message(role="assistant", content="Indeed it is!"),
        ],
    )

    # Make request to the endpoint
    response = requests.post(f"{base_url}/api/scoring", json=test_request.model_dump())

    # Assert response
    assert response.status_code == 200
    assert "score" in response.json()
    assert isinstance(response.json()["score"], float)
    assert 0 <= response.json()["score"] <= 1


def test_scoring_endpoint_validation_errors(base_url):
    # Test cases for different validation scenarios
    test_cases = [
        # Empty messages
        (
            {
                "original_messages": [],
                "compressed_messages": [],
            },
            "Message list cannot be empty",
        ),
        # Odd number of messages
        (
            {
                "original_messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                    {"role": "user", "content": "How are you?"},
                ],
                "compressed_messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
            },
            "Message list must have an even number of messages",
        ),
        # Incorrect role pattern
        (
            {
                "original_messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "user", "content": "Hi again"},
                ],
                "compressed_messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
            },
            "Messages must alternate between 'user' and 'assistant' roles",
        ),
    ]

    for test_request, expected_error in test_cases:
        response = requests.post(f"{base_url}/api/scoring", json=test_request)
        assert response.status_code == 422
        assert expected_error in str(response.json())


def test_scoring_endpoint_invalid_request(base_url):
    # Test with invalid request format
    response = requests.post(f"{base_url}/api/scoring", json={"invalid": "data"})
    assert response.status_code == 422
