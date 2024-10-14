import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date 
import os
import argparse
import sys
import time

today = date.today()
current_date = today.strftime("%y%m%d")
#print("Current date =", current_date)

now = datetime.now()
current_time = now.strftime("%H%M%S")
#print("Current Time =", current_time)

#Setup
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # Choose number of Channels, 1 = mono, 2 seperate ones?
RATE = 44100              # 44.1kHz sampling rate
CHUNK = 1024              # 2^10 samples for buffer size
REFERENCE_FILENAME = "reference.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Specify the name of the desired audio interface
#SelectedInterface = 'USB Audio CODEC '  
SelectedInterface = 'MacBook Pro-Mikrofon'

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
    fft_data = np.fft.rfft(audio_data)
    fft_magnitude = np.abs(fft_data)  # Get the magnitude of the FFT

    # Frequency axis (x-axis)
    positive_frequencies = np.fft.rfftfreq(len(audio_data), 1.0 / sample_rate)

    # Convert magnitude to dB (adding a small constant to avoid log(0))
    positive_magnitude_db = 20 * np.log10(fft_magnitude + 1e-10)

    return positive_frequencies, positive_magnitude_db

def moving_average(xdata, ydata, window_size):
    
    smoothed_ydata = np.convolve(ydata, np.ones(window_size) / window_size, mode='valid')
    smoothed_xdata = xdata[(window_size - 1) // 2 : -(window_size - 1) // 2]
    return smoothed_xdata, smoothed_ydata

def record_sample(device_index, RECORD_SECONDS, OUTPUT_FILENAME):
    # Start Recording
    stream = audio.open(format=FORMAT, input_device_index = device_index, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print(f"Recording for {RECORD_SECONDS} seconds")

    audio_frames = []
    total_frames = int(RATE / CHUNK * RECORD_SECONDS)

    for _ in range(total_frames):
        data = stream.read(CHUNK)
        audio_frames.append(data)

    print("Finished recording.")

    # Stop Recording
    stream.stop_stream()
    stream.close()
    

    # Save the recording as a WAV file
    wavefile = wave.open(OUTPUT_FILENAME, 'wb')
    wavefile.setnchannels(CHANNELS)
    wavefile.setsampwidth(audio.get_sample_size(FORMAT))
    wavefile.setframerate(RATE)
    wavefile.writeframes(b''.join(audio_frames))
    wavefile.close()

    return b''.join(audio_frames)  # Return recording in bytes

def split_audio(audio_data, sample_width, num_channels, part_duration, rate, base_filename):
    part_length = int(rate * part_duration * num_channels * sample_width)  # Number of bytes / part
    num_parts = len(audio_data) // part_length

    for i in range(num_parts):
        part_data = audio_data[i * part_length: (i + 1) * part_length]
        part_filename = f"{base_filename}_part_{i + 1}.wav"

        # Save each part as a separate WAV file
        part_wavefile = wave.open(part_filename, 'wb')
        part_wavefile.setnchannels(num_channels)
        part_wavefile.setsampwidth(sample_width)
        part_wavefile.setframerate(rate)
        part_wavefile.writeframes(part_data)
        part_wavefile.close()

        print(f"Saved part {i + 1} as {part_filename}")

def plot_fft_comparison(recorded_filename, reference_filename, window_size=17):

    #recorded
    positive_frequencies, positive_magnitude_db = Audio_fft(recorded_filename)
    ref_positive_frequencies, ref_positive_magnitude_db = Audio_fft(reference_filename)

    # Smoothed
    window_size = 17
    smoothed_frequencies, smoothed_magnitude_db = moving_average(positive_frequencies, positive_magnitude_db, window_size)
    smoothed_ref_frequencies, smoothed_ref_magnitude_db = moving_average(ref_positive_frequencies, ref_positive_magnitude_db, window_size)

    #Plot1 whole spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(positive_frequencies, positive_magnitude_db, label="Recorded")
    plt.plot(ref_positive_frequencies, ref_positive_magnitude_db, label="Reference")
    plt.plot(smoothed_frequencies, smoothed_magnitude_db, label="Smoothed Recorded", linestyle='--')
    plt.plot(smoothed_ref_frequencies, smoothed_ref_magnitude_db, label="Smoothed Reference", linestyle='--')
    plt.title("Entire Frequency Spectrum (-20 khz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DB")
    plt.legend()
    plt.grid()
    plt.xlim(0, 20000)
    plt.savefig(f"EntireSpectrum_{current_date}_{current_time}.png")
    plt.show()

    #Plot2 Spectrum (-1000)
    plt.figure(figsize=(10, 6))
    plt.plot(positive_frequencies, positive_magnitude_db, label="Recorded")
    plt.plot(ref_positive_frequencies, ref_positive_magnitude_db, label="Reference")
    plt.plot(smoothed_frequencies, smoothed_magnitude_db, label="Smoothed Recorded", linestyle='--')
    plt.plot(smoothed_ref_frequencies, smoothed_ref_magnitude_db, label="Smoothed Reference", linestyle='--')
    plt.title("Frequency Spectrum (-1 khz) ")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DB")
    plt.legend()
    plt.grid()
    plt.xlim(0, 1000)
    plt.savefig(f"Spectrum1000_{current_date}_{current_time}.png")
    plt.show()

     #Plot3 Spectrum (-200)
    plt.figure(figsize=(10, 6))
    plt.plot(positive_frequencies, positive_magnitude_db, label="Recorded")
    plt.plot(ref_positive_frequencies, ref_positive_magnitude_db, label="Reference")
    plt.plot(smoothed_frequencies, smoothed_magnitude_db, label="Smoothed Recorded", linestyle='--')
    plt.plot(smoothed_ref_frequencies, smoothed_ref_magnitude_db, label="Smoothed Reference", linestyle='--')
    plt.title("Frequency Spectrum (-200hz) ")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DB")
    plt.legend()
    plt.grid()
    plt.xlim(0, 200)
    plt.savefig(f"Spectrum200_{current_date}_{current_time}.png")
    plt.show()

    #Plot4 Difference (-1000)
    plt.figure(figsize=(10, 6))
    plt.plot(positive_frequencies, positive_magnitude_db-ref_positive_magnitude_db, label="diff")
    plt.plot(smoothed_frequencies, smoothed_magnitude_db-smoothed_ref_magnitude_db, label="diff_smooth")
    plt.title("Frequency Difference recorded - reference (-1000hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DB")
    plt.legend()
    plt.grid()
    plt.xlim(0, 1000)  
    plt.savefig('Difference(-1000).png')
    plt.show()

    #Plot5 Difference (-200)
    plt.figure(figsize=(10, 6))
    plt.plot(positive_frequencies, positive_magnitude_db-ref_positive_magnitude_db, label="diff")
    plt.plot(smoothed_frequencies, smoothed_magnitude_db-smoothed_ref_magnitude_db, label="diff_smooth")
    plt.title("Frequency Difference recorded - reference (-200hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("DB")
    plt.legend()
    plt.grid()
    plt.xlim(0, 200)  
    plt.savefig('Difference(-200).png')
    plt.show()
    

def main():

    parser = argparse.ArgumentParser(description="Choose Recording-Type")
    parser.add_argument('--duration', type=float, required= True, help="Specify duration of record in sec.")
    parser.add_argument('--split', action='store_true', help="If set, recording will be split into 0.5sec-parts")
    parser.add_argument('--continuous', action='store_true', help="If set, record in intervals for long durations.")
    parser.add_argument('--total_time', type=float, required='--continuous' in sys.argv, help="Total time to record in hours.")
    parser.add_argument('--interval', type=float, required='--continuous' in sys.argv, help="Intervals between recordings in minutes.")
    args = parser.parse_args()

    now = datetime.now()
    current_date = now.strftime("%Y%m%d")
    current_time = now.strftime("%H%M%S")

    # Get the device index for the target interface
    device_index = get_device_index_by_name(SelectedInterface)

    if device_index is None:
        print(f"Error: Your Interface '{SelectedInterface}' not found.")
    else:
        print(f"Using device: {SelectedInterface} (ID: {device_index})")


        if args.continuous:
            total_time_in_seconds = args.total_time * 3600
            interval_in_seconds = args.interval * 60
            start_time = time.time()
            elapsed_time = 0

            folder_name = f"Continuousrecording_{current_date}_{current_time}"
            os.makedirs(folder_name, exist_ok=True)

            while elapsed_time < total_time_in_seconds:
                start_record_time = time.time()
                OUTPUT_FILENAME = f"{folder_name}/Recording_{args.duration}s_{current_date}_{current_time}.wav"

                audio_data = record_sample(device_index, args.duration, OUTPUT_FILENAME)

                if args.split:
                    sample_width = audio.get_sample_size(FORMAT)
                    split_audio(audio_data, sample_width, CHANNELS, 0.5, RATE, OUTPUT_FILENAME)

                elapsed_time = time.time() - start_time
                if elapsed_time < total_time_in_seconds:
                    recording_duration = time.time() - start_record_time
                    wait_time = interval_in_seconds - recording_duration
                    if wait_time >0:
                        print(f"Waiting for {args.interval} minutes before next recording.")
                        time.sleep(wait_time)

        else: 
            RECORD_SECONDS = args.duration
            OUTPUT_FILENAME = f"Recording_{RECORD_SECONDS}s_{current_date}_{current_time}.wav"
            audio_data = record_sample(device_index, RECORD_SECONDS, OUTPUT_FILENAME)
            print("Recording finished")

    if args.split:
            OUTPUT_DIR = F"Splits_{current_date}_{current_time}"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"Splitting the {RECORD_SECONDS}s Recording into 0.5s parts")
            sample_width = audio.get_sample_size(FORMAT)
            num_channels = CHANNELS
            split_audio(audio_data, sample_width, num_channels, 0.5, RATE, f"{OUTPUT_DIR}/Recording_{current_date}_{current_time}")


    if os.path.exists(REFERENCE_FILENAME): 
        print(f"Comparing recording with reference file: {REFERENCE_FILENAME}")
        plot_fft_comparison(OUTPUT_FILENAME, REFERENCE_FILENAME)
    else:
        print(f"Reference file {REFERENCE_FILENAME} not found. Skipping comparison.")



# Ensure that PyAudio is terminated after execution
def cleanup():
    audio.terminate()

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
