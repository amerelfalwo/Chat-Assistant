from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.middleware.exp_handler import exp_handler
from server.routes.uploadpdf import router as upload_router
from server.routes.ask import router as ask_router


app = FastAPI(title="Medical Assistant", description="A medical assistant API built with FastAPI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.middleware("https")(exp_handler)

app.include_router(upload_router)
app.include_router(ask_router)