import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
import queue
import sys

# Audio configuration
SAMPLE_RATE = 44100  # Sampling rate in Hz
CHUNK_SIZE = 2048    # Number of audio samples per frame

# Create a queue to hold audio data between the audio callback and the main thread
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This is called for each audio block by sounddevice."""
    if status:
        print(f"Status: {status}", file=sys.stderr)
    # Copy the audio data and put it in the queue
    audio_queue.put(indata.copy()[:, 0])

# --- Setup matplotlib figure ---
fig, ax = plt.subplots(figsize=(10, 6))

# X-axis represents frequencies
# rfftfreq computes the frequencies for the real FFT output
xf = np.fft.rfftfreq(CHUNK_SIZE, 1 / SAMPLE_RATE)

# Initialize line for plotting (y-axis data is zeros initially)
line, = ax.plot(xf, np.zeros(len(xf)), color='cyan', lw=1.5)

# Configure plot axes and labels
ax.set_xlim(20, SAMPLE_RATE / 2) # Human hearing range roughly starts around 20Hz
ax.set_ylim(0, 0.5)              # You may need to adjust this depending on your microphone volume
ax.set_xscale('log')             # Logarithmic scale for better perception of frequencies

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude Magnitude')
ax.set_title('Real-Time Audio Spectrum Analyzer using FFT')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Apply a dark theme for a modern look
fig.patch.set_facecolor('#1e1e1e')
ax.set_facecolor('#1e1e1e')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.title.set_color('white')
for spine in ax.spines.values():
    spine.set_edgecolor('gray')

# Pre-calculate a Hanning window to reduce spectral leakage
window = np.hanning(CHUNK_SIZE)

def update_plot(frame):
    """Update function for matplotlib animation. Computes FFT and updates the plot."""
    # Process all available data in the queue and keep only the latest chunk
    data = None
    while not audio_queue.empty():
        data = audio_queue.get()
        
    if data is not None and len(data) == CHUNK_SIZE:
        # Apply windowing function
        windowed_data = data * window
        
        # Calculate Fast Fourier Transform (FFT) for real-valued input
        yf = np.fft.rfft(windowed_data)
        
        # Calculate magnitude (amplitude) of the frequencies
        magnitude = np.abs(yf) * 2 / CHUNK_SIZE
        
        # Update the plot line with the new magnitudes
        line.set_ydata(magnitude)
        
    return line,

# --- Main execution ---
if __name__ == '__main__':
    try:
        print("Starting real-time spectrum analyzer...")
        print("Please ensure your microphone is connected and not blocked by privacy settings.")
        print("Close the plot window to exit.")
        
        # Initialize and start the audio input stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        )
        
        with stream:
            # Start the matplotlib animation
            ani = animation.FuncAnimation(
                fig, update_plot, interval=30, blit=True, cache_frame_data=False
            )
            plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
