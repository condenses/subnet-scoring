from .test_server import (
    test_scoring_endpoint,
    test_scoring_endpoint_invalid_request,
    test_scoring_endpoint_empty_messages,
)


def main():
    test_scoring_endpoint()
    test_scoring_endpoint_invalid_request()
    test_scoring_endpoint_empty_messages()
