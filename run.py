"""Convenience entry point:  python run.py   (equivalent to `uvicorn app.main:app`)."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
