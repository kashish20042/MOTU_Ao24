import numpy as np
import pyaudio
from scipy.signal import square, sawtooth
from numpy import hamming

consonant = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'y', 'z', 'th', 'dh', 'sh', 'zh', 'ng']
phoneme_map = {'b': 'B', 'c': 'CH', 'd': 'D', 'f': 'F', 'g': 'G', 'h': 'H', 'j': 'J', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'p': 'P', 'r': 'R', 's': 'S', 't': 'T', 'v': 'V', 'w': 'W', 'y': 'Y', 'z': 'Z', 'th': 'TH', 'dh': 'DH', 'sh': 'SH', 'zh': 'ZH', 'ng': 'NG'}
phoneme_to_freq = {'B': 100, 'CH': 400, 'D': 100, 'F': 400, 'G': 100, 'H': 400, 'J': 400, 'K': 100, 'L': 400, 'M': 400, 'N': 400, 'P': 100, 'R': 400, 'S': 400, 'T': 100, 'V': 400, 'W': 400, 'Y': 400, 'Z': 400, 'TH': 300, 'DH': 300, 'SH': 300, 'ZH': 300, 'NG': 60}
phoneme_to_modulation = {'B': 30, 'CH': 0, 'D': 30, 'F': 0, 'G': 30, 'H': 0, 'J': 8, 'K': 0, 'L': 30, 'M': 8, 'N': 8, 'P': 0, 'R': 30, 'S': 0, 'T': 0, 'V': 8, 'W': 8, 'Y': 0, 'Z': 8, 'TH': 0, 'DH': 8, 'SH': 0, 'ZH': 8, 'NG': 8}
phoneme_to_duration = {'B': 300, 'CH': 300, 'D': 300, 'F': 300, 'G': 300, 'H': 60, 'J': 300, 'K': 300, 'L': 300, 'M': 60, 'N': 60, 'P': 300, 'R': 300, 'S': 300, 'T': 300, 'V': 300, 'W': 60, 'Y': 60, 'Z': 300, 'TH': 400, 'DH': 400, 'SH': 400, 'ZH': 400, 'NG': 400}
phoneme_to_channels = {
    'B': [5, 6, 11, 12],
    'CH': [1, 6, 7, 12],
    'D': [15, 16, 21, 22],
    'F': [6, 12, 18, 24],
    'G': [1, 2, 7, 8],
    'H': [4, 5, 10, 11,16,17,22,23],
    'J': [1, 6, 7, 12],
    'K': [1, 2, 7, 8],
    'L': [17, 18, 23, 24],
    'M': [5, 6, 11, 12],
    'N': [15, 16, 21, 22],
    'P': [5, 6, 11, 12],
    'R': [13, 14, 19, 20],
    'S': [1, 7, 13, 19],
    'T': [15, 16, 21, 22],
    'V': [6, 12, 18, 24],
    'W': [3, 4, 5, 6, 15, 16, 17, 18],
    'Y': [3, 4, 5, 6, 15, 16, 17, 18],
    'Z': [1, 7, 13, 19],
    'TH': [3, 4, 9, 10],
    'DH': [3, 4, 9, 10],
    'SH': [17, 18, 23, 24],
    'ZH': [13, 14, 19, 20],
    'NG': [1, 2, 7, 8]
}

def get_valid_consonant():
    while True:
        letter = input('Enter a consonant: ').lower()
        if letter in consonant:
            return letter
        else:
            print('Invalid input. Please enter a valid consonant.')

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

letter = get_valid_consonant()
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

print(f'Generated wave for consonant {letter} on channels {channels}')
