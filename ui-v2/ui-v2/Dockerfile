FROM python:3.10-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install fastapi uvicorn[standard] jinja2 httpx python-multipart
COPY app_ui.py /app/
COPY templates /app/templates
EXPOSE 9090
CMD ["uvicorn", "app_ui:app", "--host", "0.0.0.0", "--port", "9090"]


