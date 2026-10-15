# pysmap CLI image.
#   docker build -t pysmap .
#   docker run --rm pysmap run examples/case.yml -o /out --ons-txt          # bundled example
#   docker run --rm -v "$PWD/caso:/data" pysmap run /data                   # your case: /data/Arq_Entrada -> /data/Arq_Saida
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    # the Numba kernel caches its compiled code; keep it writable for any --user
    NUMBA_CACHE_DIR=/tmp/numba

# dependencies first, so code edits do not reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY examples ./examples
COPY tests/data ./tests/data
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["pysmap"]
CMD ["--help"]
