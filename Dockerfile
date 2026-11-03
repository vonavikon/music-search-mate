FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

# deno: JS runtime для yt-dlp (n-challenge). Минимум 2.3.0 — младшие
# помечаются unsupported и форматы молча деградируют до 403-клиентов
RUN curl -fsSL -o /tmp/deno.zip https://github.com/denoland/deno/releases/download/v2.9.5/deno-x86_64-unknown-linux-gnu.zip \
    && unzip -q /tmp/deno.zip -d /tmp/deno \
    && mv /tmp/deno/deno /usr/local/bin/deno \
    && chmod +x /usr/local/bin/deno \
    && rm -rf /tmp/deno /tmp/deno.zip

WORKDIR /app

COPY requirements.txt .
# yt-dlp: overwrite stable with the nightly build (fixes the 403 wave of
# 2026-08-17 on datacenter IPs). --upgrade is required or pip keeps the
# installed stable as "already satisfied".
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps --pre --upgrade "yt-dlp" \
    --extra-index-url https://github.com/yt-dlp/yt-dlp/releases/expanded_assets/nightly

COPY . .

RUN mkdir -p data temp logs

CMD ["python", "-m", "bot.main"]
