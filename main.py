from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import router

app = FastAPI(
    title = "SHL RECOMMENDATION",
    description="To generate suitable SHL assessment recommendations",
    version="1.0.0",
    docs_url="/docs",
)
app.add_middleware(
    CORSMiddleware,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.include_router(router.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
