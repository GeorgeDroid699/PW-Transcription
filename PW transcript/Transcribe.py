#pls run this command first pip install -r requirements.txt
#note: removed functionality for pyannote as it require HF token which im sure non of yall have/bother to create one (we use fully offline diarization)
#fully tested on python 3.11.0, running on 3.13 may result on dependacy issues
#ignore all  RuntimeWarning: divide by zero encountered in matmul mel_spec = self.mel_filters @ magnitudes this is a known issue with torchaudio and does not affect the output
#benchmark results on my mac(m4 pro chip using cpu with int8 quantization) 3min 34s for a 19min 26s audio file
#cuda is untested, run at your own risk
#changelog: code to support multithreading laptop will run hot but its faster


import os
import torch
import json
import torchaudio
import numpy as np
from tqdm import tqdm
from tempfile import mkdtemp
from datetime import timedelta
from sklearn.cluster import KMeans
from resemblyzer import VoiceEncoder, preprocess_wav
from faster_whisper import WhisperModel
from pydub import AudioSegment

# === CONFIGURATION ===
AUDIO_PATH = "/Users/vincent/Downloads/8 Jurong West St 52.mp3"
MODEL_SIZE = "large-v3"
BEST_OF = 5
BEAM_SIZE = 5
LOGPROB_THRESHOLD = -1.2
OUTPUT_PATH = "final_whisper_transcript.json"
SAMPLE_RATE = 16000
N_SPEAKERS = 2  # set manually for now

# === Device Selection ===
if torch.backends.mps.is_available():
    device = "cpu"  # ctranslate2 does not support MPS
    compute_type = "int8"
elif torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"
else:
    device = "cpu"
    compute_type = "int8"
print(f"Using device: {device}")

# === Load Models ===
print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type, cpu_threads=12)  # match to your CPU core count
encoder = VoiceEncoder()

# === Convert MP3 to WAV ===
audio = AudioSegment.from_file(AUDIO_PATH)
audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
temp_dir = mkdtemp()
temp_wav = os.path.join(temp_dir, "converted.wav")
audio.export(temp_wav, format="wav")

# === Transcribe Full Audio ===
print("Transcribing full audio with best-of sampling...")
segments, _ = model.transcribe(
    temp_wav,
    beam_size=1,
    best_of=BEST_OF,
    temperature=0.0,
    word_timestamps=True
)

# === Load waveform ===
waveform, sr = torchaudio.load(temp_wav)
waveform = waveform.squeeze().numpy()

# === Extract Embeddings from Segments ===
print("Embedding segments for speaker clustering...")
segment_embeddings = []
segment_start_times = []
segment_texts = []
segment_avg_logprobs = []

for seg in tqdm(segments, desc="Embedding segments"):
    if seg.end - seg.start < 0.1:
        continue
    start_sample = int(seg.start * SAMPLE_RATE)
    end_sample = int(seg.end * SAMPLE_RATE)
    segment_wav = waveform[start_sample:end_sample]
    if segment_wav.size == 0 or np.all(segment_wav == 0):
        continue
    wav_pre = preprocess_wav(segment_wav)
    embed = encoder.embed_utterance(wav_pre)

    segment_embeddings.append(embed)
    segment_start_times.append(seg.start)
    segment_texts.append(seg.text.strip())
    segment_avg_logprobs.append(seg.avg_logprob)

# === Speaker Clustering ===
print("Clustering speakers...")
segment_embeddings = np.vstack(segment_embeddings)
labels = KMeans(n_clusters=N_SPEAKERS, random_state=0).fit_predict(segment_embeddings)

# === Retry Low-Confidence Segments ===
print("Retrying low-confidence segments if needed...")
final_transcript = []
for i in tqdm(range(len(segment_texts)), desc="Finalizing transcript"):
    avg_logprob = segment_avg_logprobs[i]
    method = "sampling"
    retry_used = False

    if avg_logprob < LOGPROB_THRESHOLD:
        retry_segments, _ = model.transcribe(
            temp_wav,
            beam_size=BEAM_SIZE,
            best_of=1,
            temperature=0.0,
            word_timestamps=False
        )
        seg_text = " ".join(s.text.strip() for s in retry_segments)
        avg_logprob = sum(s.avg_logprob for s in retry_segments) / len(retry_segments)
        method = "beam_search"
        retry_used = True
    else:
        seg_text = segment_texts[i]

    final_transcript.append({
        "start_time": str(timedelta(seconds=int(segment_start_times[i]))),
        "speaker": f"Speaker_{labels[i]}",
        "text": seg_text,
        "avg_logprob": round(avg_logprob, 3),
        "retry_used": retry_used,
        "method": method
    })

# === Save Transcript ===
print("Saving transcript to JSON...")
with open(OUTPUT_PATH, "w") as f:
    json.dump(final_transcript, f, indent=2)
print(f"✅ Done. Transcript saved to {OUTPUT_PATH}")
