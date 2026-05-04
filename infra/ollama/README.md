# Ollama GPU Server

Run one Ollama container per GPU on a Linux server, so different models can serve from different GPUs simultaneously and the eval runner can target whichever container it wants on a per-run basis.

## How it works

`docker-compose.yml` defines one service per GPU (`ollama-gpu0` … `ollama-gpu7`), each:

- pinned to a single GPU via `deploy.resources.devices.device_ids`
- gated by a docker compose **profile** (`gpu0`, `gpu1`, …) so nothing starts unless you ask for it
- given its own model volume (`ollama-models-N`) so concurrent pulls into different containers can't collide
- attached to the shared `llm-evals` docker network — reachable from the eval container as `ollama-gpuN:11434`
- mapped to a unique host port (`11434 + N`) for ad-hoc curl/testing from the host

Pulling a model into one container only puts it on that container's volume. To serve the same model from two GPUs you pull it into both.

## One-time server setup

Prerequisites: Docker + NVIDIA GPU drivers on a Linux server. Install the NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Quick start (single GPU)

From this directory:

```bash
make up GPUS=0                          # start ollama-gpu0
make pull GPU=0 MODEL=llama3            # download llama3 into it
make gpu-check GPU=0                    # confirm GPU is visible inside the container
make test GPU=0                         # quick sanity check (asks llama3 to say hello in Welsh)
```

Then from the repo root, run an eval against it:

```bash
make eval MODEL=ollama/llama3 OLLAMA=ollama-gpu0
```

## Multi-GPU: different models on different GPUs in parallel

Bring up two (or more) GPU containers in one shot:

```bash
make up GPUS="0 3"
```

Pull each model into the container that should serve it:

```bash
make pull GPU=0 MODEL=llama3
make pull GPU=3 MODEL=gemma4:26b
```

Then run two evals in parallel — each in its own terminal — by pointing each at the right container:

```bash
# Terminal 1 — llama3 on GPU 0
make eval MODEL=ollama/llama3 OLLAMA=ollama-gpu0
```

```bash
# Terminal 2 — gemma4:26b on GPU 3
make eval MODEL=ollama/gemma4:26b OLLAMA=ollama-gpu3
```

The two eval containers don't contend: they hit different Ollama instances pinned to different GPUs.

## NVLinked pair (one model split across two GPUs)

Use this when a model is too big for one card and you want to share its layers across two GPUs (NVLink, if present, makes the inter-GPU traffic fast).

In `docker-compose.yml`, uncomment the `ollama-gpu01` example service and the matching `ollama-models-01:` volume entry. Adjust `device_ids` to the pair you want to use. Then:

```bash
docker compose --profile gpu01 up -d
docker exec ollama-gpu01 ollama pull llama3:70b
```

Run an eval against it:

```bash
make eval MODEL=ollama/llama3:70b OLLAMA=ollama-gpu01
```

You can mix and match: an `ollama-gpu01` running a 70B model alongside an `ollama-gpu3` running a smaller model, both serving evals at the same time.

## Configuration: `OLLAMA_API_BASE`

The eval runner reads `OLLAMA_API_BASE` to find Ollama. There are two ways to set it:

1. **Per-run via `OLLAMA=`** (recommended for the multi-container setup). The top-level Makefile turns `OLLAMA=ollama-gpu3` into `-e OLLAMA_API_BASE=http://ollama-gpu3:11434` for that run only.
2. **Default in `openai.env`** at the repo root, e.g. `OLLAMA_API_BASE=http://ollama-gpu0:11434`. Used when `OLLAMA=` is omitted.

When evals will hit different containers, prefer (1) and leave `openai.env` pointing at whichever GPU you treat as the default.

## Host port map

Each GPU container's host port is `11434 + N`, so you can curl them directly:

| Container       | In-network DNS         | Host port |
| --------------- | ---------------------- | --------- |
| `ollama-gpu0`   | `ollama-gpu0:11434`    | `11434`   |
| `ollama-gpu1`   | `ollama-gpu1:11434`    | `11435`   |
| `ollama-gpu2`   | `ollama-gpu2:11434`    | `11436`   |
| `ollama-gpu3`   | `ollama-gpu3:11434`    | `11437`   |
| `ollama-gpu4`   | `ollama-gpu4:11434`    | `11438`   |
| `ollama-gpu5`   | `ollama-gpu5:11434`    | `11439`   |
| `ollama-gpu6`   | `ollama-gpu6:11434`    | `11440`   |
| `ollama-gpu7`   | `ollama-gpu7:11434`    | `11441`   |

The eval container always uses the in-network DNS form. Host ports are just for poking at containers from outside docker.

## Makefile reference

`GPUS` is a space-separated list (`make up GPUS="0 3"`); `GPU` is a single index (`make pull GPU=3 MODEL=...`). Single-target commands default `GPU` to the first entry of `GPUS`.

```
make up GPUS="0 3"             Start ollama-gpu0 and ollama-gpu3
make down GPUS="0 3"           Stop only those two
make down-all                  Stop every ollama-gpu* container in this project

make pull GPU=3 MODEL=gemma4:26b
make list GPU=3                Show models in one container
make list-all                  Show models across all running ollama-gpu* containers

make gpu-check GPU=3           Run nvidia-smi inside ollama-gpu3
make test GPU=3                Quick liveness check (assumes llama3 is pulled there)
make logs GPU=3                Tail logs for ollama-gpu3
make clean                     Stop everything and delete every model volume
```

## Migrating from the old single-container setup

The earlier compose defined a single `ollama` service with a `ollama-models` volume. The new layout uses different container/volume names, so the old container and its downloaded models won't be picked up automatically.

```bash
docker compose down -v        # remove old `ollama` container AND its model volume
make up GPUS=3                # bring up the new ollama-gpu3 container
make pull GPU=3 MODEL=gemma4:26b   # re-pull whatever you had
```

If you'd rather keep the old volume's contents, run `docker volume ls` to confirm the old volume name (likely `ollama_ollama-models`), then either copy its contents into one of the new `ollama_ollama-models-N` volumes with a one-shot helper container, or just re-pull — usually faster.

## Model recommendations for Welsh evals

| Model         | VRAM needed | Notes                                              |
| ------------- | ----------- | -------------------------------------------------- |
| `llama3.2:1b` | ~1.3 GB     | Very fast, useful for checking the pipeline works  |
| `llama3.2:3b` | ~2.5 GB     | Better quality, still small                        |
| `llama3` (8B) | ~5 GB       | Good balance of speed and quality                  |
| `mistral` (7B)| ~5 GB       | Good multilingual support                          |
| `gemma2:27b`  | ~18 GB      | Strong multilingual                                |
| `gemma4:26b`  | ~17 GB      | Newer Gemma family, fits on a single A6000 (48 GB) |
