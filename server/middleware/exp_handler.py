from fastapi import  Request
from fastapi.responses import JSONResponse
from server.logger import logger


async def exp_handler(request: Request, next_handler):
    try:
        response = await next_handler(request)
        return response
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "An internal server error occurred."})