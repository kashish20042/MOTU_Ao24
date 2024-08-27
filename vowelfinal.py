import numpy as np
import pyaudio
from scipy.signal import square, sawtooth
from numpy import hamming

consonant= ['ee', 'ah', 'oo', 'ae', 'aw', 'er', 'uh', 'ih', 'eh', 'uu', 'ay', 'i', 'ow', 'oe', 'oy']
phoneme_map = {'ee': 'EE', 'ah': 'AH', 'oo': 'OO', 'ae': 'AE', 'aw': 'AW', 'er': 'ER', 'uh': 'UH', 'ih': 'IH', 'eh': 'EH', 'uu': 'UU', 'ay': 'AY', 'i': 'I', 'ow': 'OW', 'oe': 'OE', 'oy': 'OY'}
phoneme_to_freq = {'ee': 300, 'ah': 60, 'oo': 300, 'ae': 300, 'aw': 300, 'er': 300, 'uh': 300, 'ih': 300, 'eh': 300, 'uu': 300, 'ay': 300, 'i': 300, 'ow': 300, 'oe': 300, 'oy': 300}
phoneme_to_modulation = {'ee': 0, 'ah': 0, 'oo': 30, 'ae': 0, 'aw': 0, 'er': 0, 'uh': 0, 'ih': 0, 'eh': 0, 'uu': 30, 'ay': 30, 'i': 30, 'ow': 0, 'oe': 0, 'oy': 0}
phoneme_to_duration = {'ee': 480, 'ah': 480, 'oo': 480, 'ae': 480, 'aw': 480, 'er': 480, 'uh': 240, 'ih': 240, 'eh': 240, 'uu': 240, 'ay': 480, 'i': 480, 'ow': 480, 'oe': 480, 'oy': 480}
phoneme_to_channels = {
    'EE': [6, 5, 4, 3, 2, 1],
    'AH': [1, 6, 1, 6],
    'OO': [3, 4, 3, 4],
    'AE': [6, 6, 6, 6],
    'AW': [1, 2, 1, 2],
    'ER': [4, 5, 4, 5],
    'UH': [1, 6, 1, 6],
    'IH': [1, 2, 1, 2],
    'EH': [5, 6, 5, 6],
    'UU': [5, 6, 5, 6],
    'AY': [3, 4, 3, 4],
    'I': [5, 6, 5, 6],
    'OW': [23, 24, 5, 6],
    'OE': [1, 2, 1, 2],
    'OY': [1, 1, 1, 1]
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
