from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io, torch

app = FastAPI(title="YOLOv11 Classification API")
_model = None
_names = None

def get_model():
    global _model, _names
    if _model is None:
        from ultralytics import YOLO
        # Carga perezosa: descarga el peso la PRIMERA vez que se use.
        _model = YOLO("yolo11n-cls.pt")  # modelo pequeño de clasificación
        _names = _model.names
    return _model, _names

@app.get("/")
def root():
    return {"status": "ok", "model": "yolo11n-cls"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    model, names = get_model()
    # Infiere; desactiva verbosidad para ahorrar recursos
    res = model(img, verbose=False)[0]
    # Probabilidades como tensor
    probs_t = getattr(res.probs, "data", res.probs)
    topk = torch.topk(probs_t, k=min(6, probs_t.numel()))
    preds = [{"label": names[int(i)], "score": float(v)} for v, i in zip(topk.values, topk.indices)]
    return JSONResponse({"predictions": preds})
