"""
app.py — EE200 Audio Fingerprinting demo app.

Three modes:
- Library (browse the indexed song database)
- Identify (upload one clip and see the full matching pipeline)
- Batch (upload many clips, get a results.csv)
"""

import io
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import librosa
import librosa.display

import fingerprint as fp

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Audio Fingerprinting",
    page_icon="🎵",
    layout="wide",
)

ACCENT = "#D85A30"
ACCENT_DIM = "#993C1D"
BG = "#0E0C0A"
CARD_BG = "#1A130E"
TEXT_DIM = "#B89B7E"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG};
    }}
    h1, h2, h3 {{
        color: #EAFBF7;
    }}
    .eyebrow {{
        color: {ACCENT};
        font-family: monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }}
    .stat-box {{
        background-color: {CARD_BG};
        border: 1px solid #1F2A27;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }}
    .stat-box .label {{
        color: {TEXT_DIM};
        font-family: monospace;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }}
    .stat-box .value {{
        color: {ACCENT};
        font-family: monospace;
        font-size: 1.4rem;
        font-weight: 700;
    }}
    .stat-box .sub {{
        color: {TEXT_DIM};
        font-family: monospace;
        font-size: 0.7rem;
    }}
    .match-card {{
        background: linear-gradient(135deg, rgba(45,212,191,0.08), rgba(45,212,191,0.02));
        border: 1px solid {ACCENT_DIM};
        border-radius: 10px;
        padding: 1.5rem 1.8rem;
    }}
    .match-card .label {{
        color: {ACCENT};
        font-family: monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }}
    .match-card .song-name {{
        color: #EAFBF7;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.2rem 0 0.4rem 0;
    }}
    .match-card .meta {{
        color: {TEXT_DIM};
        font-family: monospace;
        font-size: 0.85rem;
    }}
    .no-match-card {{
        background: rgba(220,38,38,0.06);
        border: 1px solid rgba(220,38,38,0.4);
        border-radius: 10px;
        padding: 1.5rem 1.8rem;
    }}
</style>
""", unsafe_allow_html=True)
# Single-track playback: pause every other <audio> element the instant one
# starts playing. Also stop all audio when a Streamlit tab is switched, since
# tab content is hidden (not removed) so a playing track would otherwise keep
# going silently in the background. Streamlit re-renders often, so this is
# wrapped in a MutationObserver to keep re-attaching listeners to new audio
# elements as they appear, rather than running once on page load.
#
# Uses components.html (not st.markdown) because st.markdown's HTML rendering
# does not reliably execute <script> tags - components.html is the documented,
# supported way to run custom JS in Streamlit.
components.html("""
<script>
(function() {
    function enforceSingleAudio() {
        const audios = window.parent.document.querySelectorAll('audio');
        audios.forEach(a => {
            if (a.dataset.singlePlayBound) return;
            a.dataset.singlePlayBound = "true";
            a.addEventListener('play', () => {
                audios.forEach(other => {
                    if (other !== a) other.pause();
                });
            });
        });
    }

    function stopAllAudio() {
        window.parent.document.querySelectorAll('audio').forEach(a => a.pause());
    }

    function bindTabButtons() {
        const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
        tabs.forEach(tab => {
            if (tab.dataset.stopAudioBound) return;
            tab.dataset.stopAudioBound = "true";
            tab.addEventListener('click', stopAllAudio);
        });
    }

    const observer = new window.parent.MutationObserver(() => {
        enforceSingleAudio();
        bindTabButtons();
    });
    observer.observe(window.parent.document.body, { childList: true, subtree: true });

    enforceSingleAudio();
    bindTabButtons();
})();
</script>
""", height=0)


# ---------------------------------------------------------------------------
# Data loading (cached so the 76MB pickle loads once per server process,
# not on every interaction)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_database():
    with open("fingerprint_db.pkl", "rb") as f:
        data = pickle.load(f)
    return data["database"], data["song_names"], data["song_peaks"], data["params"]


@st.cache_resource
def load_song_metadata():
    """Artist + trivia per song, keyed by exact song name. Optional — if the
    file is missing, the app still works, just without this extra info."""
    import json
    try:
        with open("song_metadata.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


database, song_names, song_peaks, params = load_database()
song_metadata = load_song_metadata()

# Minimum top-offset score to count as a genuine match rather than chance hash
# collisions. Below this, both Identify and Batch report no match. Tune this
# against your own robustness-sweep data - wrong songs in testing scored
# roughly 3-9, true matches scored in the hundreds to thousands, so this sits
# in the ambiguous zone deliberately on the cautious side.
CONFIDENCE_THRESHOLD = 10
SR = params.get("SR", fp.SR)


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

def fig_to_none(fig):
    """Close a matplotlib figure after Streamlit has rendered it, to avoid
    memory creeping up across repeated interactions in the same session."""
    plt.close(fig)


def make_constellation_thumb(peaks, figsize=(3, 1.6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)
    if len(peaks) > 0:
        ax.scatter(peaks[:, 1], peaks[:, 0], s=1, c=ACCENT, alpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig


def make_spectrogram_and_constellation(DB, peaks, sr, hop_length, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)

    img = librosa.display.specshow(
        DB, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=axes[0]
    )
    axes[0].set_ylim(0, 5000)
    axes[0].set_title("Spectrogram", color="#EAFBF7", fontsize=10)
    axes[0].tick_params(colors=TEXT_DIM, labelsize=7)
    axes[0].xaxis.label.set_color(TEXT_DIM)
    axes[0].yaxis.label.set_color(TEXT_DIM)

    if len(peaks) > 0:
        axes[1].scatter(peaks[:, 1], peaks[:, 0], s=4, c=ACCENT)
    axes[1].set_title(f"Constellation ({len(peaks)} peaks)", color="#EAFBF7", fontsize=10)
    axes[1].set_xlabel("time frame", color=TEXT_DIM, fontsize=8)
    axes[1].set_ylabel("freq bin", color=TEXT_DIM, fontsize=8)
    axes[1].tick_params(colors=TEXT_DIM, labelsize=7)
    for spine in axes[1].spines.values():
        spine.set_color("#1F2A27")

    fig.tight_layout()
    return fig


def make_offset_histogram(offset_counts, best_offset, song_name=None):
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=BG)
    ax.set_facecolor(BG)

    raw_offsets = []
    for offset, count in offset_counts.items():
        raw_offsets.extend([offset] * count)

    n_bins = min(150, max(10, len(offset_counts)))
    ax.hist(raw_offsets, bins=n_bins, color=ACCENT_DIM, edgecolor=ACCENT, linewidth=0.3)
    ax.axvline(best_offset, color="#378ADD", linestyle="--", linewidth=1.5,
               label=f"matched offset = {best_offset}")

    best_count = offset_counts[best_offset]
    ax.annotate(f"{best_count} hashes\nalign here",
                xy=(best_offset, best_count),
                xytext=(best_offset, best_count * 0.7),
                color="#378ADD", fontsize=9, ha="center")

    ax.set_xlabel("time offset (database frame − query frame)", color=TEXT_DIM, fontsize=8)
    ax.set_ylabel("# hashes", color=TEXT_DIM, fontsize=8)
    ax.set_title("Offset histogram", color="#EAFBF7", fontsize=10)
    ax.tick_params(colors=TEXT_DIM, labelsize=7)
    ax.legend(facecolor=CARD_BG, edgecolor="#1F2A27", labelcolor="#EAFBF7", fontsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1F2A27")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(f"""
<div style="display:flex; align-items:center; gap:0.6rem;">
    <span style="font-size:1.8rem;">🎙️</span>
    <h1 style="margin:0;">Audio<span style="color:{ACCENT};"> Fingerprinting</span></h1>
</div>
<div class="eyebrow">EE200 COURSE PROJECT</div>
<p style="color:{TEXT_DIM}; margin-top:0.4rem;">
    Index a library of songs as spectrogram fingerprints, then identify any short clip against it.
</p>
""", unsafe_allow_html=True)

tab_library, tab_identify, tab_batch = st.tabs(["◆ Library", "○ Identify", "▤ Batch"])


# ---------------------------------------------------------------------------
# Library tab
# ---------------------------------------------------------------------------

with tab_library:
    st.markdown('<div class="eyebrow">LIBRARY</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color:{CARD_BG}; border:1px solid #1F2A27; border-radius:8px;
                padding:1rem; color:{TEXT_DIM}; font-family:monospace; font-size:0.85rem;
                text-align:center; margin-bottom:1.2rem;">
        Song indexing is pre-computed and shipped with this app.<br/>
        {len(song_names)} tracks · {len(database)} total fingerprint hashes
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">IN THE DATABASE</div>', unsafe_allow_html=True)

    cols_per_row = 4
    for row_start in range(0, len(song_names), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, song_id in zip(cols, range(row_start, min(row_start + cols_per_row, len(song_names)))):
            with col:
                peaks = song_peaks[song_id]
                fig = make_constellation_thumb(peaks)
                st.pyplot(fig, use_container_width=True)
                fig_to_none(fig)

                artist = song_metadata.get(song_names[song_id], {}).get("artist", "")
                artist_line = f'<div style="color:{ACCENT}; font-size:0.7rem;">{artist}</div>' if artist else ""

                card_html = (
                    '<div style="margin-top:-0.6rem;">'
                    f'<div style="color:#EAFBF7; font-size:0.85rem; font-weight:600;">{song_names[song_id]}</div>'
                    f'{artist_line}'
                    f'<div style="color:{TEXT_DIM}; font-family:monospace; font-size:0.75rem;">{len(peaks):,} peaks</div>'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Identify tab
# ---------------------------------------------------------------------------

def run_identify_pipeline(y, sr, on_stage=None):
    """
    Runs the full pipeline with per-stage timing, mirroring the demo
    video's 5-stage breakdown: spectrogram, constellation, hashing,
    db lookup, scoring.

    If `on_stage` is given, it's called as on_stage(stage_index, stage_name)
    after each of the 5 stages completes, so a caller can drive a progress bar.
    """
    timings = {}
    stage_names = ["spectrogram", "constellation", "hashing", "db lookup", "scoring"]

    t0 = time.perf_counter()
    DB = fp.compute_spectrogram(y, sr, n_fft=params.get("n_fft", fp.N_FFT),
                                 hop_length=params.get("hop_length", fp.HOP_LENGTH))
    timings["spectrogram"] = (time.perf_counter() - t0) * 1000
    if on_stage:
        on_stage(1, stage_names[0])

    t0 = time.perf_counter()
    peaks = fp.find_peaks(DB, neighborhood=params.get("neighborhood", fp.NEIGHBORHOOD),
                           threshold_db=params.get("threshold_db", fp.THRESHOLD_DB))
    timings["constellation"] = (time.perf_counter() - t0) * 1000
    if on_stage:
        on_stage(2, stage_names[1])

    t0 = time.perf_counter()
    query_hashes = fp.generate_hashes(peaks, fan_out=params.get("fan_out", fp.FAN_OUT),
                                       time_delta_max=params.get("time_delta_max", fp.TIME_DELTA_MAX))
    timings["hashing"] = (time.perf_counter() - t0) * 1000
    if on_stage:
        on_stage(3, stage_names[2])

    t0 = time.perf_counter()
    offset_counts = {}
    for h, qt in query_hashes:
        if h in database:
            for song_id, dt in database[h]:
                offset_counts.setdefault(song_id, {})
                offset_counts[song_id][dt - qt] = offset_counts[song_id].get(dt - qt, 0) + 1
    timings["db_lookup"] = (time.perf_counter() - t0) * 1000
    if on_stage:
        on_stage(4, stage_names[3])

    t0 = time.perf_counter()
    scores = {sid: max(counts.values()) for sid, counts in offset_counts.items()}
    best_id = max(scores, key=scores.get) if scores else None
    timings["scoring"] = (time.perf_counter() - t0) * 1000
    if on_stage:
        on_stage(5, stage_names[4])

    return {
        "DB": DB, "peaks": peaks, "hashes": query_hashes,
        "offset_counts": offset_counts, "scores": scores,
        "best_id": best_id,
        "best_name": song_names[best_id] if best_id is not None else None,
        "timings": timings,
    }


with tab_identify:
    st.markdown('<div class="eyebrow">SEARCH</div>', unsafe_allow_html=True)
    st.markdown("### Identify a clip")

    uploaded = st.file_uploader(
        "Upload a clip", type=["wav", "mp3", "flac", "ogg", "m4a"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")

    identify_clicked = st.button("Identify", type="primary", disabled=uploaded is None)

    if identify_clicked and uploaded is not None:
        progress = st.progress(0, text=f"Uploading {uploaded.name} ...")

        uploaded.seek(0)
        y = fp.load_audio_from_bytes(uploaded.read(), sr=SR)
        progress.progress(0.05, text="Loaded audio, starting pipeline ...")

        TOTAL_STAGES = 5
        def update_progress(stage_idx, stage_name):
            frac = 0.05 + 0.95 * (stage_idx / TOTAL_STAGES)
            progress.progress(frac, text=f"Identifying {uploaded.name} — {stage_name} ({stage_idx}/{TOTAL_STAGES})")

        result = run_identify_pipeline(y, SR, on_stage=update_progress)
        progress.empty()

        # --- timing strip ---------------------------------------------------
        t = result["timings"]
        total_ms = sum(t.values())
        stage_labels = [
            ("①", "SPECTROGRAM", f"{t['spectrogram']:.0f} ms", f"{result['DB'].shape[0]}×{result['DB'].shape[1]}"),
            ("②", "CONSTELLATION", f"{t['constellation']:.0f} ms", f"{len(result['peaks'])} peaks"),
            ("③", "HASHING", f"{t['hashing']:.0f} ms", f"{len(result['hashes']):,} hashes"),
            ("④", "DB LOOKUP", f"{t['db_lookup']:.0f} ms", f"{len(song_names)} tracks"),
            ("⑤", "SCORING", f"{t['scoring']:.0f} ms",
             f"offset {max(result['offset_counts'].get(result['best_id'], {0: 0}), key=result['offset_counts'].get(result['best_id'], {0: 0}).get) if result['best_id'] is not None else '—'}"),
        ]
        stat_cols = st.columns(len(stage_labels) + 1)
        for i, (num, label, val, sub) in enumerate(stage_labels):
            with stat_cols[i]:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="label">{num} {label}</div>
                    <div class="value">{val}</div>
                    <div class="sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)
        with stat_cols[len(stage_labels)]:
            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:center; height:100%;
                        color:{ACCENT}; font-family:monospace; font-size:0.9rem;">
                total {total_ms:.0f} ms
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # --- match card ------------------------------------------------------
        top_score = result["scores"].get(result["best_id"], 0) if result["best_id"] is not None else 0
        if result["best_id"] is None or top_score < CONFIDENCE_THRESHOLD:
            st.markdown(f"""
            <div class="no-match-card">
                <div class="label" style="color:#F87171;">NO MATCH FOUND</div>
                <div class="song-name" style="color:#F87171;">Clip not recognised</div>
                <div class="meta">No fingerprint hashes from this clip aligned with any indexed track at a consistent offset.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            best_id = result["best_id"]
            best_score = result["scores"][best_id]
            ranked = sorted(result["scores"].items(), key=lambda x: -x[1])
            runner_up_score = ranked[1][1] if len(ranked) > 1 else 0
            ratio = best_score / runner_up_score if runner_up_score > 0 else float("inf")

            meta = song_metadata.get(result["best_name"], {})
            artist = meta.get("artist")
            trivia = meta.get("trivia")

            artist_html = f'<div class="meta" style="margin-top:0.2rem;">{artist}</div>' if artist else ""

            match_card_html = (
                '<div class="match-card">'
                '<div class="label">MATCH FOUND</div>'
                f'<div class="song-name">{result["best_name"]}</div>'
                f'{artist_html}'
                f'<div class="meta" style="margin-top:0.4rem;">cluster score '
                f'<span style="color:{ACCENT};">{best_score}</span>'
                f' · {ratio:.0f}x the runner-up</div>'
                '</div>'
            )
            st.markdown(match_card_html, unsafe_allow_html=True)

            if trivia:
                st.markdown(f"""
                <div style="background-color:{CARD_BG}; border:1px solid #2A1F18; border-radius:8px;
                            padding:0.9rem 1.1rem; margin-top:0.6rem; color:{TEXT_DIM};
                            font-size:0.85rem; line-height:1.5;">
                    <span style="color:{ACCENT}; font-family:monospace; font-size:0.7rem;
                                  letter-spacing:0.08em; text-transform:uppercase;">trivia</span><br/>
                    {trivia}
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="eyebrow">CANDIDATE SCORES</div>', unsafe_allow_html=True)
            max_score = ranked[0][1] if ranked else 1
            for sid, score in ranked[:5]:
                pct = int(100 * score / max_score) if max_score > 0 else 0
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.35rem;">
                    <div style="width:220px; color:#EAFBF7; font-size:0.85rem;">{song_names[sid]}</div>
                    <div style="flex:1; background:#1F2A27; border-radius:4px; height:10px; overflow:hidden;">
                        <div style="width:{pct}%; background:{ACCENT}; height:100%;"></div>
                    </div>
                    <div style="width:50px; text-align:right; color:{TEXT_DIM}; font-family:monospace; font-size:0.8rem;">{score}</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="eyebrow">STEP 1 · FEATURE EXTRACTION</div>', unsafe_allow_html=True)
            st.markdown("**From spectrogram to constellation**")
            fig1 = make_spectrogram_and_constellation(
                result["DB"], result["peaks"], SR, params.get("hop_length", fp.HOP_LENGTH)
            )
            st.pyplot(fig1, use_container_width=True)
            fig_to_none(fig1)

            st.write("")
            st.markdown('<div class="eyebrow">STEP 2 · DATABASE SEARCH</div>', unsafe_allow_html=True)
            st.markdown("**Where in the song?**")
            st.markdown(f"""
            <p style="color:{TEXT_DIM}; font-size:0.85rem;">
                The {len(result['hashes']):,} fingerprint hashes from this clip were looked up against
                every indexed track. <b style="color:{ACCENT};">{best_score}</b> of them agreed on a single
                time offset within <b>{result['best_name']}</b>.
            </p>
            """, unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="eyebrow">STEP 3 · THE PROOF</div>', unsafe_allow_html=True)
            st.markdown("**The alignment spike**")
            best_offset = max(result["offset_counts"][best_id], key=result["offset_counts"][best_id].get)
            fig2 = make_offset_histogram(result["offset_counts"][best_id], best_offset, result["best_name"])
            st.pyplot(fig2, use_container_width=True)
            fig_to_none(fig2)
            st.markdown(f"""
            <p style="color:{TEXT_DIM}; font-size:0.85rem;">
                Every matched hash votes for a time offset (database frame minus query frame). Chance
                matches scatter their votes randomly across many offsets, forming a flat noise floor.
                A genuine match makes them converge: <b style="color:#378ADD;">{best_score} hashes agreed
                on a single offset</b>. That spike cannot be a coincidence.
            </p>
            """, unsafe_allow_html=True)



# ---------------------------------------------------------------------------
# Batch tab
# ---------------------------------------------------------------------------

with tab_batch:
    st.markdown('<div class="eyebrow">BATCH</div>', unsafe_allow_html=True)
    st.markdown("### Identify many clips at once")
    st.markdown(f"""
    <p style="color:{TEXT_DIM}; font-size:0.85rem;">
        Upload a set of query clips. Each is identified against the currently indexed library,
        and the results are written to a standardised <code>results.csv</code> with columns
        <code>filename, prediction</code>. <code>prediction</code> is the matched track's filename
        without its extension, or <code>none</code> when no candidate clears the confidence threshold.
    </p>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload clips", type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if batch_files:
        st.markdown('<div class="eyebrow">UPLOADED CLIPS — CLICK TO PLAY</div>', unsafe_allow_html=True)
        for file in batch_files:
            play_col, name_col = st.columns([1, 5])
            with play_col:
                file.seek(0)
                st.audio(file, format=f"audio/{file.name.split('.')[-1]}")
            with name_col:
                st.markdown(f"""
                <div style="display:flex; align-items:center; height:100%; color:{TEXT_DIM};
                            font-family:monospace; font-size:0.85rem;">
                    {file.name}
                </div>
                """, unsafe_allow_html=True)
        st.write("")

    run_batch = st.button("Run batch", type="primary", disabled=not batch_files)

    if run_batch and batch_files:
        upload_status = st.empty()
        upload_status.markdown(f"""
        <div style="color:{TEXT_DIM}; font-family:monospace; font-size:0.85rem;">
            ⬆ uploading {len(batch_files)} file(s) ...
        </div>
        """, unsafe_allow_html=True)

        progress = st.progress(0, text=f"Identifying {batch_files[0].name} ... (1/{len(batch_files)})")
        upload_status.empty()

        rows = []

        for i, file in enumerate(batch_files):
            progress.progress(i / len(batch_files),
                               text=f"Identifying {file.name} ... ({i+1}/{len(batch_files)})")

            file.seek(0)
            y = fp.load_audio_from_bytes(file.read(), sr=SR)
            result = fp.identify(y, SR, database, song_names, params)

            best_id = result["best_id"]
            top_score = result["scores"].get(best_id, 0) if best_id is not None else 0

            if best_id is not None and top_score >= CONFIDENCE_THRESHOLD:
                prediction = song_names[best_id]
            else:
                prediction = "none"

            rows.append({"filename": file.name, "prediction": prediction})
            progress.progress((i + 1) / len(batch_files),
                               text=f"Identifying {file.name} ... ({i+1}/{len(batch_files)}) done")

        progress.empty()

        import pandas as pd
        results_df = pd.DataFrame(rows, columns=["filename", "prediction"])

        st.markdown('<div class="eyebrow">RESULTS</div>', unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results.csv", data=csv_bytes,
            file_name="results.csv", mime="text/csv",
        )