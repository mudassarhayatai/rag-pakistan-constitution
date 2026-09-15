"""
HTTP client for the FastAPI backend, used by the Streamlit frontend
(app.py).
"""

import requests

DEFAULT_TIMEOUT = 60


class APIError(Exception):
    """Raised for anything the UI should show as a user-facing error
    message rather than letting an exception propagate raw."""


def check_health(base_url: str) -> dict:
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise APIError(f"Could not reach the API at {base_url} -- is it running?") from e


def ask_question(base_url: str, question: str, candidate_k: int | None = None, top_k: int | None = None) -> dict:
    payload = {"question": question}
    if candidate_k is not None:
        payload["candidate_k"] = candidate_k
    if top_k is not None:
        payload["top_k"] = top_k

    try:
        resp = requests.post(f"{base_url}/query", json=payload, timeout=DEFAULT_TIMEOUT)
    except requests.exceptions.ConnectionError as e:
        raise APIError(f"Could not reach the API at {base_url} -- is it running?") from e
    except requests.exceptions.Timeout as e:
        raise APIError(f"Request timed out after {DEFAULT_TIMEOUT}s -- the LLM may be slow to respond") from e

    if resp.status_code in (404, 502, 503):
        detail = resp.json().get("detail", resp.text) if resp.content else resp.reason
        raise APIError(detail)
    resp.raise_for_status()
    return resp.json()