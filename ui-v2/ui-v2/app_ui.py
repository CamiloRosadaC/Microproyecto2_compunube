import os
import httpx
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

app = FastAPI(title="MP2 - UI v2")
templates = Jinja2Templates(directory="templates")

# Bases por ENV (en cluster usaremos los Service DNS)
YOLO_BASE  = os.getenv("YOLO_BASE",  "http://yolo-cls-svc.mp2.svc.cluster.local")
HELLO_BASE = os.getenv("HELLO_BASE", "http://hello-svc.mp2.svc.cluster.local")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/hello-proxy")
async def hello_proxy():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{HELLO_BASE}/")  # HELLO_BASE = http://hello-svc.mp2.svc.cluster.local
        return Response(content=r.content, status_code=r.status_code, media_type="text/html")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": repr(e)})

@app.get("/ping-yolo")
async def ping_yolo():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{YOLO_BASE}/")
        return {"ok": r.status_code, "text": r.text[:200], "base": YOLO_BASE}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": repr(e), "base": YOLO_BASE})

@app.get("/ping-hello")
async def ping_hello():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{HELLO_BASE}/")
        return {"ok": r.status_code, "text": r.text[:200], "base": HELLO_BASE}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": repr(e), "base": HELLO_BASE})

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            r = await client.post(f"{YOLO_BASE}/predict", files=files)
        if r.status_code // 100 != 2:
            return JSONResponse(
                status_code=r.status_code,
                content={"error": "Upstream error",
                         "status": r.status_code,
                         "text": r.text[:1000],
                         "base": YOLO_BASE}
            )
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": repr(e), "base": YOLO_BASE})

