"""
fingerprint.py

Parameters here MUST match the ones used to build fingerprint_db.pkl,
since hashes are exact-match lookups (no tolerance for drift). The
pickled database stores its own copy of these params under 'params',
which app.py reads to keep them in sync.
"""

import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from collections import Counter, defaultdict

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
NEIGHBORHOOD = 20
THRESHOLD_DB = -30
FAN_OUT = 15
TIME_DELTA_MAX = 200


def compute_spectrogram(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Compute a dB-scaled magnitude spectrogram via STFT."""
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    return DB


def find_peaks(DB, neighborhood=NEIGHBORHOOD, threshold_db=THRESHOLD_DB):
    """
    Find local maxima in the spectrogram that stand out from their
    surroundings and exceed an absolute loudness threshold.
    Returns an array of [freq_bin, time_frame] rows.
    """
    local_max = maximum_filter(DB, size=neighborhood) == DB
    above_thresh = DB > threshold_db
    peak_mask = local_max & above_thresh
    return np.argwhere(peak_mask)


def generate_hashes(peaks, fan_out=FAN_OUT, time_delta_max=TIME_DELTA_MAX):
    """
    Pair each peak with up to `fan_out` nearby future peaks to form
    hashes of the form (f1, f2, dt). Returns a list of (hash, anchor_time).
    """
    hashes = []
    peaks_sorted = peaks[peaks[:, 1].argsort()]
    for i, (f1, t1) in enumerate(peaks_sorted):
        for j in range(1, fan_out + 1):
            if i + j >= len(peaks_sorted):
                break
            f2, t2 = peaks_sorted[i + j]
            dt = int(t2) - int(t1)
            if dt <= 0 or dt > time_delta_max:
                continue
            h = (int(f1), int(f2), dt)
            hashes.append((h, int(t1)))
    return hashes


def generate_single_peak_hashes(peaks):
    """Single-peak hash variant: just the frequency bin, no pairing."""
    return [((int(f),), int(t)) for f, t in peaks]


def fingerprint_audio(y, sr, params=None):
    """
    Run the full pipeline (spectrogram -> peaks -> hashes) on a raw
    audio array. Returns (DB, peaks, hashes) so callers can reuse the
    intermediate steps for visualization without recomputing them.
    """
    p = params or {}
    DB = compute_spectrogram(
        y, sr,
        n_fft=p.get('n_fft', N_FFT),
        hop_length=p.get('hop_length', HOP_LENGTH),
    )
    peaks = find_peaks(
        DB,
        neighborhood=p.get('neighborhood', NEIGHBORHOOD),
        threshold_db=p.get('threshold_db', THRESHOLD_DB),
    )
    hashes = generate_hashes(
        peaks,
        fan_out=p.get('fan_out', FAN_OUT),
        time_delta_max=p.get('time_delta_max', TIME_DELTA_MAX),
    )
    return DB, peaks, hashes


def identify(y, sr, database, song_names, params=None):
    """
    Identify a query audio array against the fingerprint database.

    Returns a dict with:
        - 'best_id': matched song_id, or None if no match found
        - 'best_name': matched song name, or None
        - 'scores': {song_id: top_offset_count} for every candidate with any hits
        - 'offset_counts': {song_id: Counter(offset -> count)} full detail,
           used for plotting the offset histogram
        - 'DB', 'peaks', 'hashes': intermediate pipeline outputs, used for
           the spectrogram / constellation visuals
    """
    DB, peaks, query_hashes = fingerprint_audio(y, sr, params)

    result = {
        'best_id': None,
        'best_name': None,
        'scores': {},
        'offset_counts': {},
        'DB': DB,
        'peaks': peaks,
        'hashes': query_hashes,
    }

    if len(query_hashes) == 0:
        return result

    offset_counts = defaultdict(Counter)
    for h, qt in query_hashes:
        if h in database:
            for song_id, dt in database[h]:
                offset_counts[song_id][dt - qt] += 1

    scores = {song_id: counter.most_common(1)[0][1]
              for song_id, counter in offset_counts.items()}

    result['offset_counts'] = dict(offset_counts)
    result['scores'] = scores

    if scores:
        best_id = max(scores, key=scores.get)
        result['best_id'] = best_id
        result['best_name'] = song_names[best_id]

    return result


def load_audio_from_bytes(file_bytes, sr=SR):
    import io
    try:
        y, _ = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True)
        return y
    except Exception:
        return _load_audio_via_ffmpeg(file_bytes, sr=sr)


def _load_audio_via_ffmpeg(file_bytes, sr=SR):
    import subprocess
    import numpy as np

    cmd = [
        "ffmpeg", "-i", "pipe:0",
        "-f", "f32le", "-ac", "1", "-ar", str(sr),
        "-loglevel", "error", "pipe:1",
    ]
    proc = subprocess.run(cmd, input=file_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode audio: {proc.stderr.decode(errors='ignore')}")
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    return y