"""FastAPI application exposing the DriftSentry model."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request

from driftsentry import __version__
from driftsentry.api.schemas import (
    DriftRequest,
    DriftResponse,
    Health,
    ModelInfo,
    ScoreRequest,
    ScoreResponse,
)
from driftsentry.api.service import InferenceService
from driftsentry.api.settings import Settings
from driftsentry.artifacts import load_bundle

log = logging.getLogger("driftsentry.api")


def create_app(service: InferenceService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            path = Path(settings.artifacts_dir)
            if (path / "meta.json").exists():
                bundle = load_bundle(path, device=settings.device)
                app.state.service = InferenceService(bundle, device=settings.device)
                log.info("loaded model bundle from %s", path)
            else:
                log.warning("no model bundle at %s; run `driftsentry train` first", path)
        yield

    app = FastAPI(title="DriftSentry", version=__version__, lifespan=lifespan)
    app.state.service = service
    app.state.settings = settings

    def get_service(request: Request) -> InferenceService:
        svc = request.app.state.service
        if svc is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return svc

    def check_size(n: int) -> None:
        if n > settings.max_transactions:
            raise HTTPException(
                status_code=413,
                detail=f"too many transactions ({n} > {settings.max_transactions})",
            )

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        return Health(status="ok", model_loaded=request.app.state.service is not None)

    @app.get("/model/info", response_model=ModelInfo)
    def model_info(svc: InferenceService = Depends(get_service)) -> dict:
        return svc.info()

    @app.post("/score", response_model=ScoreResponse, response_model_exclude_none=True)
    def score(req: ScoreRequest, svc: InferenceService = Depends(get_service)) -> dict:
        check_size(len(req.transactions))
        txns = [t.model_dump() for t in req.transactions]
        return svc.score(txns, include_windows=req.include_windows)

    @app.post("/drift", response_model=DriftResponse)
    def drift(req: DriftRequest, svc: InferenceService = Depends(get_service)) -> dict:
        check_size(len(req.transactions))
        txns = [t.model_dump() for t in req.transactions]
        return svc.drift(txns, include_features=req.include_features)

    return app


app = create_app()
