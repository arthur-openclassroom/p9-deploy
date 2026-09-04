FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1

WORKDIR /app

COPY api/requirements.txt /app/api/requirements.txt
# Wheel CPU de torch : image beaucoup plus legere que le paquet par defaut (CUDA inclus)
RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /app/api/requirements.txt

COPY api /app/api
COPY models/segformer_b0_best.pt /app/models/segformer_b0_best.pt

ENV MODEL_PATH=/app/models/segformer_b0_best.pt
# 7860 : port attendu par Hugging Face Spaces (app_port dans l'en-tete du README).
ENV PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
