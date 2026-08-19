from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.db.database import init_db
from backend.api import forecast_routes, reorder_routes, agent_routes, audit_routes

from backend.config import ALLOWED_ORIGINS

app = FastAPI(title="Retail Demand Forecasting API")

# Setup CORS middleware with explicit allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Initialize the database on application startup."""
    init_db()

# Include all API routers
app.include_router(forecast_routes.router, prefix="/forecast", tags=["Forecast"])
app.include_router(reorder_routes.router, prefix="/reorder", tags=["Reorder"])
app.include_router(agent_routes.router, prefix="/agent", tags=["Agent"])
app.include_router(audit_routes.router, prefix="/audit", tags=["Audit"])

@app.get("/")
def read_root():
    return {"message": "Retail Demand Forecasting API is running"}
