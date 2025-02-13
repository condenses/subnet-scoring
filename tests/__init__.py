from .test_server import (
    test_scoring_endpoint,
    test_scoring_endpoint_validation_errors,
    test_scoring_endpoint_invalid_request,
)


def main():
    test_scoring_endpoint()
    test_scoring_endpoint_validation_errors()
    test_scoring_endpoint_invalid_request()
