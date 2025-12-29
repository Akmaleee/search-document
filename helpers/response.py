from fastapi.responses import JSONResponse

def response(message: str, data=None, error=None, status_code: int = 200):
    """Helper untuk membuat respons API dengan struktur konsisten."""
    return JSONResponse(
        status_code=status_code,
        content={
            "message": message,
            "data": data,
            "error": error
        }
    )