# SentrySearch

Semantic search over dashcam footage. Type what you're looking for, get a trimmed clip back.

[<video src="https://github.com/ssrajadh/sentrysearch/raw/main/docs/demo.mp4" controls width="100%"></video>](https://github.com/user-attachments/assets/da83db09-887b-44e4-a04b-118de57e1700)

## How it works

sentrysearch splits your dashcam videos into overlapping chunks, embeds each chunk directly as video using Google's Gemini Embedding model, and stores the vectors in a local ChromaDB database. When you search, your text query is embedded into the same vector space and matched against the stored video embeddings. The top match is automatically trimmed from the original file and saved as a clip.

## Getting Started

1. Clone and install:

```bash
git clone https://github.com/ssrajadh/sentrysearch.git
cd sentrysearch
python -m venv venv && source venv/bin/activate
pip install -e .
```

2. Set up your API key:

```bash
sentrysearch init
```

This prompts for your Gemini API key, writes it to `.env`, and validates it with a test embedding.

3. Index your footage:

```bash
sentrysearch index /path/to/dashcam/footage
```

4. Search:

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
$ sentrysearch index /path/to/dashcam/footage
Indexing file 1/3: front_2024-01-15_14-30.mp4 [chunk 1/4]
Indexing file 1/3: front_2024-01-15_14-30.mp4 [chunk 2/4]
...
Indexed 12 new chunks from 3 files. Total: 12 chunks from 3 files.
```

Options: `--chunk-duration 30` (seconds per chunk), `--overlap 5` (overlap between chunks).

### Search

```bash
$ sentrysearch search "red truck running a stop sign"
  #1 [0.87] front_2024-01-15_14-30.mp4 @ 02:15-02:45
  #2 [0.74] left_2024-01-15_14-30.mp4 @ 02:10-02:40
  #3 [0.61] front_2024-01-20_09-15.mp4 @ 00:30-01:00

Saved clip: ./match_front_2024-01-15_14-30_02m15s-02m45s.mp4
```

Options: `--results N`, `--output-dir DIR`, `--no-trim` to skip auto-trimming.

### Stats

```bash
$ sentrysearch stats
Total chunks:  47
Source files:  12
```

### Verbose mode

Add `--verbose` to either command for debug info (embedding dimensions, API response times, similarity scores).

## How is this possible?

Gemini Embedding 2 can natively embed video — raw video pixels are projected into the same 768-dimensional vector space as text queries. There's no transcription, no frame captioning, no text middleman. A text query like "red truck at a stop sign" is directly comparable to a 30-second video clip at the vector level. This is what makes sub-second semantic search over hours of footage practical.

## Cost

Indexing 1 hour of footage costs ~$2.50 with Gemini's embedding API. This is with default settings (30s chunks, 5s overlap). Costs can be reduced by increasing chunk duration or decreasing overlap. Search queries are negligible (text embedding only).

## Limitations & Future Work

- **Indexing cost can be optimized** — currently every chunk is embedded regardless of content. Skipping still frames, reducing frame rate before embedding, or using smarter chunking (scene detection) could significantly reduce API calls and cost.
- **Search quality depends on chunk boundaries** — if an event spans two chunks, the overlapping window helps but isn't perfect.
- **Gemini Embedding 2 is in preview** — API behavior and pricing may change.

## Compatibility

This works with any footage in mp4 format, not just Tesla Sentry Mode. The directory scanner recursively finds all `.mp4` files regardless of folder structure.

## Requirements

- Python 3.10+
- `ffmpeg` on PATH, or use bundled ffmpeg via `imageio-ffmpeg` (installed by default)
- Gemini API key ([get one free](https://aistudio.google.com/apikey))
