
import uvicorn

from src.config import get_settings
from src.main import init

if __name__ == "__main__":
    init()
    uvicorn.run("src.main:app", host="0.0.0.0", port=get_settings().fastapi_port, reload=True)
