# SentrySearch

Semantic search over video footage. Type what you're looking for, get a trimmed clip back.

[OpenClaw Skill](https://clawhub.ai/ssrajadh/natural-language-video-search)

[<video src="https://github.com/ssrajadh/sentrysearch/raw/main/docs/demo.mp4" controls width="100%"></video>](https://github.com/user-attachments/assets/baf98fad-080b-48e1-97f5-a2db2cbd53f5)

## How it works

SentrySearch splits your mp4 videos into overlapping chunks, embeds each chunk as video using either Google's Gemini Embedding API or a local Qwen3-VL model, and stores the vectors in a local ChromaDB database. When you search, your text query is embedded into the same vector space and matched against the stored video embeddings. The top match is automatically trimmed from the original file and saved as a clip.

## Getting Started

1. Install [uv](https://docs.astral.sh/uv/) (if you don't have it):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # macOS/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```

2. Clone and install:

```bash
git clone https://github.com/ssrajadh/sentrysearch.git
cd sentrysearch
uv sync
```

3. Set up your API key:

```bash
sentrysearch init
```

This prompts for your Gemini API key, writes it to `.env`, and validates it with a test embedding.

4. Index your footage:

```bash
sentrysearch index /path/to/dashcam/footage
```

5. Search:

```bash
sentrysearch search "red truck running a stop sign"
```

ffmpeg is required for video chunking and trimming. If you don't have it system-wide, the bundled `imageio-ffmpeg` is used automatically.

> **Manual setup:** If you prefer not to use `sentrysearch init`, you can copy `.env.example` to `.env` and add your key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) manually.

## Usage

### Init

```bash
$ sentrysearch init
Enter your Gemini API key (get one at https://aistudio.google.com/apikey): ****
Validating API key...
Setup complete. You're ready to go — run `sentrysearch index <directory>` to get started.
```

If a key is already configured, you'll be asked whether to overwrite it.

### Index footage

```bash
$ sentrysearch index /path/to/video/footage
Indexing file 1/3: front_2024-01-15_14-30.mp4 [chunk 1/4]
Indexing file 1/3: front_2024-01-15_14-30.mp4 [chunk 2/4]
...
Indexed 12 new chunks from 3 files. Total: 12 chunks from 3 files.
```

Options:

- `--chunk-duration 30` — seconds per chunk
- `--overlap 5` — overlap between chunks
- `--no-preprocess` — skip downscaling/frame rate reduction (send raw chunks)
- `--target-resolution 480` — target height in pixels for preprocessing
- `--target-fps 5` — target frame rate for preprocessing
- `--no-skip-still` — embed all chunks, even ones with no visual change

### Search

```bash
$ sentrysearch search "red truck running a stop sign"
  #1 [0.87] front_2024-01-15_14-30.mp4 @ 02:15-02:45
  #2 [0.74] left_2024-01-15_14-30.mp4 @ 02:10-02:40
  #3 [0.61] front_2024-01-20_09-15.mp4 @ 00:30-01:00

Saved clip: ./match_front_2024-01-15_14-30_02m15s-02m45s.mp4
```

If the best result's similarity score is below the confidence threshold (default 0.41), you'll be prompted before trimming:

```
No confident match found (best score: 0.28). Show results anyway? [y/N]:
```

With `--no-trim`, low-confidence results are shown with a note instead of a prompt.

Options: `--results N`, `--output-dir DIR`, `--no-trim` to skip auto-trimming, `--threshold 0.5` to adjust the confidence cutoff.

### Local Backend (no API key needed)

Index and search using a local Qwen3-VL-Embedding model instead of the Gemini API. Free, private, and runs entirely on your machine.

The default model is **Qwen3-VL-Embedding-8B**. Pick an install based on your hardware:

| Hardware | Install command | What happens |
|---|---|---|
| **Apple Silicon, 32 GB+** (M1/M2/M3/M4/M5 Pro/Max) | `uv sync --extra local` | Full float16 precision via MPS (~16 GB unified memory) |
| **NVIDIA, 18 GB+ VRAM** (A100, RTX 3090/4090) | `uv sync --extra local` | Full bf16 precision (~18 GB VRAM) |
| **NVIDIA, 8–16 GB VRAM** (most consumer GPUs) | `uv sync --extra local-quantized` | 4-bit quantization via bitsandbytes (~6–8 GB VRAM) |

> **Not sure?** On Mac, use `--extra local`. On NVIDIA, use `--extra local-quantized` — 4-bit quantization works on the widest range of NVIDIA hardware with minimal quality loss. (bitsandbytes requires CUDA and does not work on Mac/MPS.)

Index and search with `--backend local`:

```bash
sentrysearch index /path/to/footage --backend local
sentrysearch search "car running a red light" --backend local
```

Options:
- `--model qwen2b` — smaller model, lower quality but only ~4 GB VRAM (also accepts full HuggingFace IDs)
- `--quantize` / `--no-quantize` — force 4-bit quantization on or off (default: auto-detect based on whether bitsandbytes is installed)

Notes:
- First run downloads the model (~16 GB for 8B, ~4 GB for 2B).
- Requires a GPU for reasonable speed (CUDA or Apple Metal). CPU works but is very slow.
- Embeddings from Gemini and local backends are **not compatible** — an index built with one backend cannot be searched with the other. Re-index if you switch backends.
- Switching models (e.g. 8B → 2B) also produces incompatible embeddings — re-index if you change models.

### Tesla Metadata Overlay

Burn speed, location, and time onto trimmed clips:

```bash
sentrysearch search "car cutting me off" --overlay
```

This extracts telemetry embedded in Tesla dashcam files (speed, GPS) and renders a HUD overlay. The overlay shows:

- **Top center:** speed and MPH label on a light gray card
- **Below card:** date and time (12-hour with AM/PM)
- **Top left:** city and road name (via reverse geocoding)

![tesla overlay](docs/tesla-overlay.png)

Requirements:

- Tesla firmware 2025.44.25 or later, HW3+
- SEI metadata is only present in driving footage (not parked/Sentry Mode)
- Reverse geocoding uses [OpenStreetMap's Nominatim API](https://nominatim.openstreetmap.org/) via geopy (optional)

Install with Tesla overlay support:

```bash
uv sync --extra tesla
```

Without geopy, the overlay still works but omits the city/road name.

Source: [teslamotors/dashcam](https://github.com/teslamotors/dashcam)

### Stats

```bash
$ sentrysearch stats
Total chunks:  47
Source files:  12
Backend:       gemini
```

### Verbose mode

Add `--verbose` to either command for debug info (embedding dimensions, API response times, similarity scores).

## How is this possible?

Both Gemini Embedding 2 and Qwen3-VL-Embedding can natively embed video — raw video pixels are projected into the same vector space as text queries. There's no transcription, no frame captioning, no text middleman. A text query like "red truck at a stop sign" is directly comparable to a 30-second video clip at the vector level. This is what makes sub-second semantic search over hours of footage practical.

## Cost

Indexing 1 hour of footage costs ~$2.84 with Gemini's embedding API (default settings: 30s chunks, 5s overlap):

> 1 hour = 3,600 seconds of video = 3,600 frames processed by the model.
> 3,600 frames × $0.00079 = ~$2.84/hr

The Gemini API natively extracts and tokenizes exactly 1 frame per second from uploaded video, regardless of the file's actual frame rate. The preprocessing step (which downscales chunks to 480p at 5fps via ffmpeg) is a local/bandwidth optimization — it keeps payload sizes small so API requests are fast and don't timeout — but does not change the number of frames the API processes.

Two built-in optimizations help reduce costs in different ways:

- **Preprocessing** (on by default) — chunks are downscaled to 480p at 5fps before uploading. Since the API processes at 1fps regardless, this only reduces upload size and transfer time, not the number of frames billed. It primarily improves speed and prevents request timeouts.
- **Still-frame skipping** (on by default) — chunks with no meaningful visual change (e.g. a parked car) are skipped entirely. This saves real API calls and directly reduces cost. The savings depend on your footage — Sentry Mode recordings with hours of idle time benefit the most, while action-packed driving footage may have nothing to skip.

Search queries are negligible (text embedding only).

Tuning options:

- `--chunk-duration` / `--overlap` — longer chunks with less overlap = fewer API calls = lower cost
- `--no-skip-still` — embed every chunk even if nothing is happening
- `--target-resolution` / `--target-fps` — adjust preprocessing quality
- `--no-preprocess` — send raw chunks to the API

## Limitations & Future Work

- **Still-frame detection is heuristic** — it uses JPEG file size comparison across sampled frames. It may occasionally skip chunks with subtle motion or embed chunks that are truly static. Disable with `--no-skip-still` if you need every chunk indexed.
- **Search quality depends on chunk boundaries** — if an event spans two chunks, the overlapping window helps but isn't perfect. Smarter chunking (e.g. scene detection) could improve this.
- **Gemini Embedding 2 is in preview** — API behavior and pricing may change.

## Compatibility

This works with any footage in mp4 format, not just Tesla Sentry Mode. The directory scanner recursively finds all `.mp4` files regardless of folder structure.

## Requirements

- Python 3.11+
- `ffmpeg` on PATH, or use bundled ffmpeg via `imageio-ffmpeg` (installed by default)
- **Gemini backend:** Gemini API key ([get one free](https://aistudio.google.com/apikey))
- **Local backend:** GPU with CUDA or Apple Metal recommended; `uv sync --extra local`
