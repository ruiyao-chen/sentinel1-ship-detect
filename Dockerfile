# Offline test image: the detector, the synthetic backend and the full pytest
# suite, no network or cloud credentials needed.
#
#   docker build -t s1ship .
#   docker run --rm s1ship                                   # runs pytest
#   docker run --rm -v "$PWD/results:/app/results" -v "$PWD/figures:/app/figures" \
#       s1ship python scripts/run_synthetic.py               # synthetic validation
#
# Build with --build-arg REAL=1 to add rasterio and earthengine-api for the
# real-data scripts (scripts/fetch_data.py, scripts/run_detect.py).
FROM python:3.12-slim

ARG REAL=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt requirements-real.txt ./
RUN pip install -r requirements.txt \
 && if [ "$REAL" = "1" ]; then pip install -r requirements-real.txt; fi

COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY data/ data/

RUN useradd --create-home app && mkdir -p results figures && chown -R app /app
USER app

CMD ["python", "-m", "pytest", "-q"]
