import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
    RAGUploadResponse,
)

from app.services.rag_service import rag_service


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


@router.post(
    "/upload",
    response_model=RAGUploadResponse
)
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    temp_path = None

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF is empty."
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        result = rag_service.ingest_pdf(
            temp_path
        )

        return RAGUploadResponse(
            message="PDF uploaded and indexed successfully.",
            pages=result["pages"],
            chunks=result["chunks"],
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post(
    "/query",
    response_model=RAGQueryResponse
)
async def query_document(
    request: RAGQueryRequest
):

    try:

        result = rag_service.ask(
            request.question
        )

        return RAGQueryResponse(
            answer=result["answer"],
            sources=result["sources"],
        )

    except RuntimeError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )