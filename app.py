from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import analyze_package, REVIEW_SAMPLE_SIZE_DEFAULT
from database import load_reviews_detailed
from urllib.parse import urlparse, parse_qs


def extract_package_id(raw: str) -> str:
    """If `raw` is a Google Play URL, extract the `id` query parameter; else return raw trimmed."""
    if not raw or not isinstance(raw, str):
        return raw
    raw = raw.strip()
    try:
        parsed = urlparse(raw)
        if parsed.scheme and parsed.netloc:
            qs = parse_qs(parsed.query)
            if 'id' in qs and qs['id']:
                return qs['id'][0]
    except Exception:
        pass
    return raw

app = FastAPI(
    title="SS-LDA Analysis API",
    description="Backend API untuk analisis review Google Play menggunakan SSLDA dan sentiment analysis.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    package_id: str = Field(..., description="Google Play package_id atau URL aplikasi")
    sample_size: int = Field(
        REVIEW_SAMPLE_SIZE_DEFAULT,
        ge=10,
        description="Jumlah review maksimum yang diproses. Ubah REVIEW_SAMPLE_SIZE_DEFAULT atau env var untuk default global.",
    )
    use_multiprocessing: bool = Field(True, description="Gunakan multiprocessing untuk klasifikasi sentimen")
    num_workers: Optional[int] = Field(None, description="Jumlah worker CPU, default menggunakan semua core")
    association_rules_file: Optional[str] = Field(
        None,
        description="Path file association rules .pkl jika ingin menggunakan file khusus",
    )


class AnalyzeResponse(BaseModel):
    package_id: str
    sample_size: int
    data_counts: dict
    models: dict


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "SS-LDA Analysis API"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        pid = extract_package_id(request.package_id)
        result = analyze_package(
            package_id=pid,
            sample_size=request.sample_size,
            use_multiprocessing=request.use_multiprocessing,
            num_workers=request.num_workers,
            association_rules_file=request.association_rules_file,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


@app.get("/reviews/{package_id}")
async def get_reviews(package_id: str, limit: Optional[int] = 50):
    try:
        reviews = load_reviews_detailed(package_id, limit=limit)
        return {"package_id": package_id, "count": len(reviews), "reviews": reviews}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
