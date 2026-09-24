"""Live spectrogram of a whistle from the laptop microphone.

Usage:
    ../le-venv/bin/python whistle-spectrogram.py              # live scrolling spectrogram
    ../le-venv/bin/python whistle-spectrogram.py --record 5   # record 5 s, then plot it

Live mode listens to the default microphone and draws a scrolling
spectrogram (time on x, frequency on y, brightness = loudness in dB).
The white line tracks the whistle's pitch: the loudest frequency in the
whistle band (WHISTLE_MIN_HZ-WHISTLE_MAX_HZ), shown only when it stands
out above the background by at least DETECT_DB. The title shows the
current pitch in Hz.

Record mode captures a fixed number of seconds, then shows a static
spectrogram of the whole clip with the pitch track on top.

Keys (live mode):
    s   save the current spectrogram to whistle-spectrogram.png
    q   quit (or close the window)
"""

import argparse
import queue

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation

SAMPLE_RATE = 44100   # Hz
FFT_SIZE = 2048       # samples per FFT window (~46 ms, ~21 Hz resolution)
HOP_SIZE = 512        # samples between successive FFT columns
HISTORY_SEC = 5       # how many seconds the live plot shows
MAX_FREQ_HZ = 6000    # top of the plot; whistles live well below this

WHISTLE_MIN_HZ = 500    # search band for the whistle pitch
WHISTLE_MAX_HZ = 5000
DETECT_DB = 20          # peak must be this far above the band's median

WINDOW = np.hanning(FFT_SIZE)
FREQS = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
PLOT_BINS = FREQS <= MAX_FREQ_HZ
BAND = (FREQS >= WHISTLE_MIN_HZ) & (FREQS <= WHISTLE_MAX_HZ)


def spectrum_db(frame):
    """Magnitude spectrum of one frame in dB, cropped to MAX_FREQ_HZ."""
    mag = np.abs(np.fft.rfft(frame * WINDOW))
    return 20 * np.log10(mag + 1e-10)[PLOT_BINS]


def whistle_pitch(column_db):
    """Loudest frequency in the whistle band, or NaN if nothing stands out."""
    band_db = column_db[BAND[PLOT_BINS]]
    peak = np.argmax(band_db)
    if band_db[peak] - np.median(band_db) < DETECT_DB:
        return np.nan
    return FREQS[BAND][peak]


def spectrogram(audio):
    """Split audio into overlapping frames and FFT each one."""
    n_frames = 1 + (len(audio) - FFT_SIZE) // HOP_SIZE
    return np.array([spectrum_db(audio[i * HOP_SIZE:i * HOP_SIZE + FFT_SIZE])
                     for i in range(n_frames)]).T


def record_and_plot(seconds):
    print(f"Recording {seconds} s... whistle now!")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    audio = audio[:, 0]
    print("Done.")

    spec = spectrogram(audio)
    times = (np.arange(spec.shape[1]) * HOP_SIZE + FFT_SIZE / 2) / SAMPLE_RATE
    pitch = np.array([whistle_pitch(col) for col in spec.T])

    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = spec.max()
    img = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                    extent=[0, times[-1], 0, MAX_FREQ_HZ],
                    vmin=vmax - 80, vmax=vmax)
    ax.plot(times, pitch, color="white", linewidth=1.5, label="whistle pitch")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Whistle spectrogram")
    ax.legend(loc="upper right")
    fig.colorbar(img, ax=ax, label="Level (dB)")
    fig.tight_layout()
    plt.show()


def live():
    audio_q = queue.Queue()

    def on_audio(indata, frames, time, status):
        audio_q.put(indata[:, 0].copy())

    n_cols = int(HISTORY_SEC * SAMPLE_RATE / HOP_SIZE)
    n_bins = int(PLOT_BINS.sum())
    spec = np.full((n_bins, n_cols), -100.0)
    pitch = np.full(n_cols, np.nan)
    buffer = np.zeros(0, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 5))
    img = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                    extent=[-HISTORY_SEC, 0, 0, MAX_FREQ_HZ],
                    vmin=-40, vmax=40)
    times = np.linspace(-HISTORY_SEC, 0, n_cols)
    (pitch_line,) = ax.plot(times, pitch, color="white", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    title = ax.set_title("Whistle into the mic...")
    fig.colorbar(img, ax=ax, label="Level (dB)")
    fig.tight_layout()

    def update(_):
        nonlocal buffer
        while not audio_q.empty():
            buffer = np.concatenate([buffer, audio_q.get()])

        new_cols = []
        while len(buffer) >= FFT_SIZE:
            new_cols.append(spectrum_db(buffer[:FFT_SIZE]))
            buffer = buffer[HOP_SIZE:]
        if not new_cols:
            return img, pitch_line, title

        new = np.array(new_cols).T
        k = new.shape[1]
        spec[:, :-k] = spec[:, k:]
        spec[:, -k:] = new
        pitch[:-k] = pitch[k:]
        pitch[-k:] = [whistle_pitch(col) for col in new.T]

        # Keep the color scale following the loudest recent sound.
        vmax = max(spec.max(), 0)
        img.set_clim(vmax - 80, vmax)
        img.set_data(spec)
        pitch_line.set_ydata(pitch)

        current = pitch[-1]
        title.set_text(f"Whistle pitch: {current:.0f} Hz" if not np.isnan(current)
                       else "Whistle into the mic...")
        return img, pitch_line, title

    def on_key(event):
        if event.key == "s":
            fig.savefig("whistle-spectrogram.png", dpi=150)
            print("Saved whistle-spectrogram.png")
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=HOP_SIZE, callback=on_audio):
        anim = FuncAnimation(fig, update, interval=30, blit=False,
                             cache_frame_data=False)
        plt.show()
    del anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=float, metavar="SECONDS",
                        help="record for this many seconds, then plot")
    args = parser.parse_args()

    if args.record:
        record_and_plot(args.record)
    else:
        live()
