import numpy as np
import pyaudio
from scipy.signal import square, sawtooth
from numpy import hamming

ragas = ['sa', 're', 'ga', 'ma', 'pa', 'dha', 'ni']
phoneme_map = {'sa': 'S', 're': 'R', 'ga': 'G', 'ma': 'M', 'pa': 'P', 'dha': 'DH', 'ni': 'N'}
phoneme_to_freq = {'S': 300, 'R': 300, 'G': 300, 'M': 300, 'P': 300, 'DH': 300, 'N': 300}
phoneme_to_modulation = {'S': 0, 'R': 30, 'G': 0, 'M': 0, 'P': 0, 'DH': 0, 'N': 60}
phoneme_to_duration = {'S': 100, 'R': 400, 'G': 400, 'M': 400, 'P': 100, 'DH': 400, 'N': 400}
phoneme_to_channels = {
    'S': [1, 7, 13, 19],
    'R': [13, 14, 19, 20],
    'G': [1, 2, 7, 8],
    'M': [5, 6, 11, 12],
    'P': [5, 6, 11, 12],
    'DH': [3, 4, 9, 10],
    'N': [15, 16, 21, 22]
}

def get_valid_raga():
    while True:
        letter = input('Enter a raga (sa, re, ga, ma, pa, dha, ni): ').lower()
        if letter in ragas:
            return letter
        else:
            print('Invalid input. Please enter a valid raga.')

def get_valid_waveform_type():
    while True:
        print('Select waveform type:')
        print('1. Sine wave')
        print('2. Square wave')
        print('3. Sawtooth wave')
        try:
            choice = int(input('Enter your choice (1/2/3): '))
            if choice in [1, 2, 3]:
                return choice
            else:
                print('Invalid selection. Please enter 1, 2, or 3.')
        except ValueError:
            print('Invalid input. Please enter a number (1, 2, or 3).')

samp_freq = 44100

letter = get_valid_raga()
type_of_wave = get_valid_waveform_type()

phoneme = phoneme_map[letter]
freq = phoneme_to_freq[phoneme]
modulation = phoneme_to_modulation[phoneme]
duration = phoneme_to_duration[phoneme] / 1000  # Convert ms to seconds
channels = phoneme_to_channels[phoneme]

time_vector = np.linspace(0, duration, int(samp_freq * duration), endpoint=False)

if type_of_wave == 1:
    waveform = np.sin(2 * np.pi * freq * time_vector)
elif type_of_wave == 2:
    waveform = square(2 * np.pi * freq * time_vector)
elif type_of_wave == 3:
    waveform = sawtooth(2 * np.pi * freq * time_vector)

# Apply modulation if needed
if modulation != 0:
    waveform *= np.sin(2 * np.pi * modulation * time_vector)
    
    # Apply Hamming window
hamming_window = hamming(len(waveform))
waveform *= hamming_window

waveform *= 0.5  # Adjust amplitude

p = pyaudio.PyAudio()

def play_wave_on_laptop(wave):
    stream = p.open(format=pyaudio.paFloat32, channels=2, rate=samp_freq, output=True)
    buffer = np.zeros((len(wave), 2))
    buffer[:, 0] = wave
    buffer[:, 1] = wave
    stream.write(buffer.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()

def play_wave_on_motu(wave, channels, device_index):
    # Initialize stream for MOTU 24Ao
    stream = p.open(format=pyaudio.paFloat32, channels=24, rate=samp_freq, output=True, output_device_index=device_index)
    buffer = np.zeros((len(wave), 24))
    for ch in channels:
        # Play the wave on the current channel
        buffer[:, ch - 1] = wave
        stream.write(buffer.astype(np.float32).tobytes())
        # Clear the buffer for the current channel before moving to the next channel
        buffer[:, ch - 1] = 0
    stream.stop_stream()
    stream.close()

play_wave_on_laptop(waveform)

# Get the device index for MOTU 24Ao
device_index = None
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if 'MOTU' in dev['name'] and dev['maxOutputChannels'] == 24:
        device_index = i
        break

if device_index is not None:
    # Play waveform on the MOTU 24Ao through the specified channels
    play_wave_on_motu(waveform, channels, device_index)
else:
    print('MOTU 24Ao device not found.')

p.terminate()

print(f'Generated wave for raga {letter} on channels {channels}')
