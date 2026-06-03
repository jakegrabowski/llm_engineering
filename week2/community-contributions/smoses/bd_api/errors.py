class BdApiError(Exception):
    """Base exception for all BdApi errors."""
    pass

class BdApiRequestError(BdApiError):
    """Exception raised when the server responds with a 4xx or 5xx status code before streaming starts."""
    def __init__(self, status_code: int, response_body: str):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(f"HTTP {status_code}: {response_body}")

class BdApiStreamError(BdApiError):
    """Exception raised when an error event is received during SSE streaming."""
    def __init__(self, message: str, code: str = None, retryable: bool = False):
        self.message = message
        self.code = code
        self.retryable = retryable
        super().__init__(f"Stream Error: {message} (code: {code}, retryable: {retryable})")
