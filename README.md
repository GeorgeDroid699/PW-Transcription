# PW-Transcription

A minimal transcription pipeline for converting a single audio file into a diarized JSON transcript. It uses [faster-whisper](https://github.com/guillaumekln/faster-whisper) and fully offline speaker diarization via `resemblyzer`.

## Getting Started

1. **Install dependencies**
   ```bash
   cd "PW transcript"
   pip install -r requirements.txt
   ```
   The script was tested with **Python 3.11**. Newer versions (e.g. 3.13) may require additional dependency fixes.

2. **Configure `Transcribe.py`**
   Edit the variables at the top of `PW transcript/Transcribe.py` to suit your environment:
   - `AUDIO_PATH` – path to the MP3 file you want to transcribe
   - `MODEL_SIZE` – Whisper model size (`large-v3` by default)
   - `BEST_OF` / `BEAM_SIZE` – sampling/beam search parameters
   - `LOGPROB_THRESHOLD` – log‑probability threshold for retrying segments
   - `OUTPUT_PATH` – where the resulting JSON file is written
   - `SAMPLE_RATE` – audio sample rate (default `16000`)
   - `N_SPEAKERS` – number of speakers to cluster

3. **Run the script**
   ```bash
   python "PW transcript/Transcribe.py"
   ```
   The script prints progress as it transcribes the audio, embeds segments and clusters speakers. When finished, a JSON transcript is saved to `OUTPUT_PATH`.

## Prompt Template for LLMs

After generating the JSON transcript you can use an LLM (e.g. ChatGPT) to produce a clean interview transcript. Below is a prompt template. Replace the speaker names with the actual names from your interview and paste the contents of the JSON file where indicated.

```
You are a professional transcription assistant.
Speaker_00 = <Speaker 1 name>
Speaker_01 = <Speaker 2 name>
Using the JSON transcript below, output a structured interview where each line is formatted as:
Speaker name: text
Do not include any additional commentary.

<PASTE JSON HERE>
```

This prompt instructs the model to rewrite the transcript with real speaker names while preserving the order of dialogue.

## Notes

- The script chooses CPU or GPU automatically. MPS (Apple Silicon) defaults to CPU because `ctranslate2` does not yet support MPS.
- Ignore any `RuntimeWarning: divide by zero` from torchaudio; it does not affect the output.
- Benchmarks on a Mac M4 Pro CPU with int8 quantization: a 19m26s file takes about **3m34s**.

