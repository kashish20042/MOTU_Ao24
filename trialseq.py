import numpy as np
import pyaudio
from scipy.signal import square, sawtooth

ragas = ['sa', 're', 'ga', 'ma', 'pa', 'dha', 'ni']
phoneme_map = {'sa': 'SA', 're': 'RE', 'ga': 'GA', 'ma': 'MA', 'pa': 'PA', 'dha': 'DHA', 'ni': 'NI'}
phoneme_to_freq = {'SA': 261.63, 'RE': 293.66, 'GA': 329.63, 'MA': 349.23, 'PA': 392.00, 'DHA': 440.00, 'NI': 493.88}
phoneme_to_amp = {'SA': 0.1, 'RE': 0.2, 'GA': 0.3, 'MA': 0.4, 'PA': 0.5, 'DHA': 0.6, 'NI': 0.7}

channels = {
    'sa': [1, 2, 3, 4],
    're': [5, 6, 7, 8],
    'ga': [9, 10, 11, 12],
    'ma': [13, 14, 15, 16],
    'pa': [17, 18, 19, 20],
    'dha': [21, 22, 23, 24],
    'ni': [21, 22, 23, 24]  
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

samp_freq = 48000  
samp_time_period = 1  
time_vector = np.linspace(0, samp_time_period, int(samp_freq * samp_time_period), endpoint=False)

letter = get_valid_raga()
type_of_wave = get_valid_waveform_type()

phoneme = phoneme_map[letter]
freq = phoneme_to_freq[phoneme]
amp = phoneme_to_amp[phoneme]

if type_of_wave == 1:
    waveform = amp * np.sin(2 * np.pi * freq * time_vector)
    waveform_type = 'Sine'
elif type_of_wave == 2:
    waveform = amp * square(2 * np.pi * freq * time_vector)
    waveform_type = 'Square'
elif type_of_wave == 3:
    waveform = amp * sawtooth(2 * np.pi * freq * time_vector)
    waveform_type = 'Sawtooth'

target_channels = channels[letter]

p = pyaudio.PyAudio()

def play_wave_on_laptop(wave):
    stream = p.open(format=pyaudio.paFloat32, channels=2, rate=samp_freq, output=True)
    buffer = np.zeros((len(wave), 2))
    buffer[:, 0] = wave
    buffer[:, 1] = wave
    stream.write(buffer.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()

def play_wave_on_motu_sequentially(wave, channels, device_index):
    # Initialize stream for MOTU 24Ao
    stream = p.open(format=pyaudio.paFloat32, channels=24, rate=samp_freq, output=True, output_device_index=device_index)
    buffer = np.zeros((len(wave), 24))
    for ch in channels:
        buffer[:, ch - 1] = wave
        stream.write(buffer.astype(np.float32).tobytes())
        buffer[:, ch - 1] = 0
        stream.write(buffer.astype(np.float32).tobytes())
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
    # Play waveform on the MOTU 24Ao sequentially
    play_wave_on_motu_sequentially(waveform, target_channels, device_index)
else:
    print('MOTU 24Ao device not found.')

p.terminate()

print(f'Generated {waveform_type} wave for raga {letter} on channels {target_channels}')
