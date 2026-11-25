
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes.query import router as query_router

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import OptiRAGError
from app.core.logging_config import configure_logging

from fastapi.responses import JSONResponse
from app.services.health_service import HealthService

configure_logging()
logger = logging.getLogger("optirag")

health_service = HealthService()

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "A production-oriented RAG system for retrieval "
        "optimization and automated evaluation."
    ),
    version="0.1.0",
    debug=settings.DEBUG,
)

app.include_router(query_router, prefix=settings.API_V1_PREFIX)
# Allow the future React frontend to communicate with the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def root():
    return {
        "project": settings.APP_NAME,
        "message": "OptiRAG backend is running.",
        "status": "ok",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return health_service.check_liveness()


@app.get("/ready", tags=["Health"])
def readiness_check():
    result = health_service.check_readiness()

    if result["status"] != "ready":
        return JSONResponse(
            status_code=503,
            content=result,
        )

    return result

@app.exception_handler(OptiRAGError)
async def optirag_exception_handler(
    request: Request,
    exc: OptiRAGError,
):
    logger.warning(
        "Application error on %s %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__,
            "detail": exc.public_message,
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected server error occurred.",
        },
    )