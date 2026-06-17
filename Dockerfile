# Faraday app container (portability artifact). The llama-servers stay NATIVE on the
# Pi (NEON-tuned build, no container RAM tax on a 4 GB board) — this image runs the
# FastAPI app anywhere, pointed at those servers. See docs/report.md for the trade-off.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV FARADAY_GEN_URL=http://host.docker.internal:8080 \
    FARADAY_EMBED_URL=http://host.docker.internal:8081 \
    FARADAY_DB=/app/data/faraday.sqlite
EXPOSE 8000
CMD ["faraday", "serve", "--host", "0.0.0.0", "--port", "8000"]
