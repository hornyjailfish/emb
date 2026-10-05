import uvicorn

from src.config import get_settings

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=get_settings().fastapi_port, reload=True)