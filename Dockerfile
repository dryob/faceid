FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.5.3 setuptools wheel cython
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# insightface pulls full opencv-python transitively; force the headless build last
RUN pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==5.0.0.93
COPY app /app/app
COPY static /app/static
COPY config.yaml /app/config.yaml
COPY faceid_nas_guard.py /app/faceid_nas_guard.py
WORKDIR /app
ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
CMD ["python", "-m", "app.main"]
