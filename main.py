
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

#Setup
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # Choose number of Channels, 1 = mono, 2 seperate ones?
RATE = 44100              # 44.1kHz sampling rate
CHUNK = 1024              # 2^10 samples for buffer size
RECORD_SECONDS = 0.1        # Recordduration
OUTPUT_FILENAME = "recording.wav"  # File name to save


# Initialize PyAudio
audio = pyaudio.PyAudio()

# Specify the name of the desired audio interface
SelectedInterface = 'USB Audio CODEC '  # 

def get_device_index_by_name(device_name):
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if (device_info.get('maxInputChannels') > 0) and (device_info.get('name') == device_name):
            return i  # Return the device index if it matches the name
    return None  # Return None if no matching device is found

def Audio_fft(wav_file):
    """
    Analyze audio frames by performing FFT and converting the magnitude to dB.

    Parameters:
    - audio_frames: list of byte strings, the audio frames to be analyzed
    - sample_rate: int, the sampling rate of the audio (e.g., 44100)

    Returns:
    - positive_frequencies: numpy array, the positive frequency components
    - positive_magnitude_db: numpy array, the magnitude in dB for the positive frequencies
    """
    # Convert the recorded byte data to a numpy array
    sample_rate, audio_data = wavfile.read(wav_file)

    # Perform FFT (Fast Fourier Transform)
    fft_data = np.fft.fft(audio_data)
    fft_magnitude = np.abs(fft_data)  # Get the magnitude of the FFT

    # Frequency axis (x-axis)
    frequencies = np.fft.fftfreq(len(fft_magnitude), 1.0 / sample_rate)

    # Only keep the positive frequencies and corresponding magnitude
    positive_frequencies = frequencies[:len(frequencies) // 2]
    positive_magnitude = fft_magnitude[:len(frequencies) // 2]

    # Convert magnitude to dB (adding a small constant to avoid log(0))
    positive_magnitude_db = 20 * np.log10(positive_magnitude + 1e-10)

    return positive_frequencies, positive_magnitude_db

def record_sample(device_index):
    # Start Recording
    stream = audio.open(format=FORMAT, input_device_index = device_index, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")

    audio_frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        audio_frames.append(data)

    print("Finished recording.")

    # Stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recording as a WAV file
    wavefile = wave.open(OUTPUT_FILENAME, 'wb')
    wavefile.setnchannels(CHANNELS)
    wavefile.setsampwidth(audio.get_sample_size(FORMAT))
    wavefile.setframerate(RATE)
    wavefile.writeframes(b''.join(audio_frames))
    wavefile.close()

def main():
    # Get the device index for the target interface
    device_index = get_device_index_by_name(SelectedInterface)

    if device_index is None:
        print(f"Error: Your Interface '{SelectedInterface}' not found.")
    else:
        print(f"Using device: {SelectedInterface} (ID: {device_index})")

    record_sample(device_index)
    print("Recording finished")
    positive_frequencies, positive_magnitude_db = Audio_fft(OUTPUT_FILENAME)

    ref_positive_frequencies, ref_positive_magnitude_db = Audio_fft("reference.wav")

    # Plot the FFT result
    plt.figure(figsize=(10, 6))
    plt.plot(positive_frequencies, positive_magnitude_db, label="rec")
    plt.plot(ref_positive_frequencies, ref_positive_magnitude_db, label="ref")
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DB")
    plt.legend()
    plt.grid()
    plt.xlim(0, RATE/2)  # Limit x-axis to Nyquist frequency
    plt.savefig('spectrum')
    #plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(positive_frequencies, positive_magnitude_db-ref_positive_magnitude_db, label="rec")
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DB")
    plt.legend()
    plt.grid()
    plt.xlim(0, RATE/2)  # Limit x-axis to Nyquist frequency
    plt.savefig('spectrum_diff')
    plt.show()

if __name__ == "__main__":
    main()