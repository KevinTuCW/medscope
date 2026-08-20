# medscope container image -- runs the FastAPI app (health probe + the
# radiologist workbench) fully offline. The workbench's sample-study path
# (medscope.bootstrap.build_sample_deps, wired in from workbench.py) is
# hermetic by construction: offline stand-ins for the VLM/arbiter/report
# stages, the real CNN reader against locally-cached weights, no network
# call and no API key required at runtime. This image never bakes in a real
# key -- see .dockerignore and the note below `pip install` for how that
# was verified -- so there is nothing to accidentally point at a live
# provider even if USE_REAL_VLM were flipped on later.
FROM python:3.12-slim

# Non-root user + its home dir, created up front so every later COPY/RUN
# that touches /app or $HOME lands with the right ownership instead of
# needing a blanket chown at the end.
RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /home/app app

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
COPY data/samples ./data/samples

# Pin torch/numpy/torchxrayvision to the exact combo verified compatible in
# development, rather than letting pip's resolver pick independently at
# build time. torch 2.2.2 against a numpy 2.x has a documented ABI mismatch
# (see src/medscope/readers/cnn.py's module docstring on why `_to_tensor`
# uses `torch.frombuffer` instead of `torch.from_numpy`) -- that workaround
# is unconditional and doesn't rely on any particular numpy being installed,
# but landing on a DIFFERENT incompatible pair here (e.g. a newer torch than
# torchxrayvision 1.5.2 supports) is still a real risk this pin closes off.
RUN pip install --no-cache-dir "numpy==1.26.4" "torch==2.2.2" "torchxrayvision==1.5.2" \
    && pip install --no-cache-dir ".[cv,llm]"
# No .env, credential, or key reaches this or any later layer: the build
# context is filtered by .dockerignore (.env/.env.*/.venv excluded, only
# .env.example allowed through), no COPY here names .env or a secret file,
# and no RUN/ENV in this file sets VLM_API_KEY or any other
# credential. Verified by building the image and running, from OUTSIDE the
# container, `docker history --no-trunc` over every layer plus
# `docker run --rm <image> find / -xdev -iname '.env*'` -- neither turns up
# a real .env or key material (see Task 3.3 report for the exact commands).

RUN chown -R app:app /app
USER app
ENV HOME=/home/app

# Pre-fetch the CNN weights (~27 MB, densenet121-res224-all) at build time
# rather than leaving them for the first /workbench/run request: a cold
# first request downloading model weights over the network is exactly the
# kind of implicit runtime network dependency this image is meant NOT to
# have. Must run as the `app` user (not root): torchxrayvision caches to
# $HOME/.torchxrayvision/models_data, and a root-owned cache would be
# invisible to the app user the CMD below actually runs as.
RUN python -c "import torchxrayvision as xrv; xrv.models.DenseNet(weights='densenet121-res224-all')"

# setuptools' `pip install ".[cv,llm]"` (a non-editable, regular install)
# leaves a source-copy build/ artifact behind in the build context's
# WORKDIR; it's not needed once the wheel is installed into site-packages.
RUN rm -rf /app/build

ENV PYTHONUNBUFFERED=1 \
    USE_REAL_VLM=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["python", "-m", "uvicorn", "medscope.app:app", "--host", "0.0.0.0", "--port", "8000"]
