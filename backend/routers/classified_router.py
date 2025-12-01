from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from services.ImageService import ImageService

app_router = APIRouter()
image_service = ImageService()

@app_router.post("/classified", status_code=status.HTTP_200_OK)
async def get_answer(
    message: str = Form(None),
    file: UploadFile = File(None)
):
    try:
        # Trường hợp: chỉ có file
        if file and not message:
            file_bytes = await file.read()
            result = await image_service.detect_image(file_bytes)
            return {
                "message": "Image processed successfully",
                "prediction": result["predicted_class"],
                "probability": result["probability"]
                # "file": result["output_image_path"]
            }
        # Trường hợp không có gì
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You must provide either a file or a message."
            )

    except Exception as e:
        print("Error:", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
