# Presentado por:
Camilo Eduardo Rosada Caicedo - 2205121

Edilmer Chachinoy Narvaez - 22501262


# Microproyecto 2 – Computación en la Nube (Minikube)

Implementación en **Minikube (2 nodos)** de:
- **YOLOv11** (clasificación de imágenes, FastAPI)
- **App de interés (Hola Mundo)** en Nginx
- **UI v2** (FastAPI + Jinja2) para:
  - Subir imagen y ver predicciones YOLO
  - Ver la página Hello embebida
  - Indicador de estado (opcional)
- **Métricas** con `metrics-server` (`kubectl top`)

## 🚀 Requisitos
- WSL2 con Ubuntu
- Docker Desktop + WSL integration
- Minikube + kubectl
- Python 3.10+ (opcional venvs para pruebas locales)

## 📂 Estructura
```
yolo-cls/
  Dockerfile
  app.py
  requirements.txt
  k8s-yolo.yaml
app-interes/
  Dockerfile
  k8s-hello.yaml
  (index.html si usas ConfigMap)
ui-v2/
  Dockerfile
  app_ui.py
  templates/index.html
  k8s-ui.yaml
k8s-ingress.yaml
```

## 1) Arranque del clúster (WSL)
```bash
minikube -p mp2 start --nodes=2 --cpus=4 --memory=8192 --disk-size=40g
minikube -p mp2 addons enable ingress
minikube -p mp2 addons enable metrics-server
kubectl create ns mp2 2>/dev/null || true
kubectl get nodes
```

## 2) Construir imágenes y desplegar (WSL)
```bash
# YOLO
cd yolo-cls
docker build -t yolo-cls:1.0 .
minikube -p mp2 image load yolo-cls:1.0
kubectl apply -f k8s-yolo.yaml

# Hello
cd ../app-interes
docker build -t app-interes:1.0 .
minikube -p mp2 image load app-interes:1.0
kubectl apply -f k8s-hello.yaml

# UI v2
cd ../ui-v2
docker build -t mp2-ui:1.0 .
minikube -p mp2 image load mp2-ui:1.0
kubectl apply -f k8s-ui.yaml
```

Verifica:
```bash
kubectl -n mp2 get deploy,svc,ing
```

## 3) Ingress (YOLO y Hello)
```bash
cd ..
kubectl apply -f k8s-ingress.yaml
```

**Windows → C:\Windows\System32\drivers\etc\hosts**
```
127.0.0.1 yolo.local hello.local ui.local
```

**Port-forward del Ingress (WSL, terminal aparte):**
```bash
kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller 8080:80 --address 0.0.0.0
```

**Pruebas:**
```bash
curl -H "Host: yolo.local"  http://127.0.0.1:8080/
curl -H "Host: hello.local" http://127.0.0.1:8080/ | head
curl -s -H "Host: yolo.local" \
  -X POST -F "file=@/mnt/c/Users/CAMILO/Downloads/dog.jpg" \
  http://127.0.0.1:8080/predict | jq
```

## 4) UI v2 para navegador (sin Ingress)
```bash
kubectl -n mp2 port-forward svc/ui-v2-svc 9090:80 --address 0.0.0.0
```

Navegador (Windows): **http://127.0.0.1:9090/**  
- Subir imagen → JSON de `predictions`  
- Panel “Hola Mundo” embebido

## 5) Métricas
```bash
kubectl top nodes
kubectl -n mp2 top pods
```

## 6) Apagado (sin borrar nada)
```bash
# Cierra port-forwards (Ctrl+C en las terminales)
minikube -p mp2 stop
```

Notas:
- Puertos **estáticos**:
  - `yolo-cls-svc` → NodePort **31111**
  - `hello-svc`   → NodePort **30080**
  - Ingress expuesto por port-forward a **8080**
  - UI v2 expuesta por port-forward a **9090**
- El primer `/predict` puede tardar por descarga/caliente del modelo.
- No se suben venvs ni artefactos pesados al repositorio.
```
