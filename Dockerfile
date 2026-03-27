FROM python:3.11-slim

# HF Spaces runs as non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Copy requirements first for layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY --chown=user . .

EXPOSE 7860

CMD ["python", "app.py"]
