import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from src.api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)
