
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import (
    RetrievalError,
    GenerationError,
)
from app.core.logging_config import configure_logging
from fastapi import Request
from fastapi.responses import JSONResponse
import logging


configure_logging()
logger = logging.getLogger("optirag.test")

app = FastAPI()


@app.exception_handler(RetrievalError)
@app.exception_handler(GenerationError)
async def application_error_handler(
    request: Request,
    exc,
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__,
            "detail": exc.public_message,
        },
    )


@app.get("/test-retrieval-error")
def trigger_retrieval_error():
    raise RetrievalError("Internal retrieval detail")


@app.get("/test-generation-error")
def trigger_generation_error():
    raise GenerationError("Internal generation detail")


client = TestClient(app)


def test_retrieval_error_response():
    response = client.get("/test-retrieval-error")

    assert response.status_code == 503
    assert response.json()["error"] == "RetrievalError"
    assert "temporarily unavailable" in response.json()["detail"]


def test_generation_error_response():
    response = client.get("/test-generation-error")

    assert response.status_code == 503
    assert response.json()["error"] == "GenerationError"
    assert "temporarily unavailable" in response.json()["detail"]
