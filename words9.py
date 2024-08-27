import numpy as np
import pyaudio
import time
from scipy.signal import square, sawtooth
from numpy import hanning

# Define phoneme mappings
phoneme_map = {
    'ACE': ['AY', 'S'], 'AID': ['AY', 'D'], 'DAY': ['D', 'AY'], 'MAY': ['M', 'AY'], 
    'ME': ['M', 'EE'], 'MOO': ['M', 'OO'], 'SUE': ['S', 'UU'], 'DOOM': ['D', 'OO', 'M'], 
    'DUDE': ['D', 'UU', 'D'], 'MOOSE': ['M', 'OO', 'S'], 'SAME': ['S', 'AY', 'M'], 
    'SEAM': ['S', 'EE', 'M'], 'SEED': ['S', 'EE', 'D'], 'KEY': ['K', 'EE'], 'MY': ['M', 'I'], 
    'SIGH': ['S', 'I'], 'THEY': ['DH', 'AY'], 'WAY': ['W', 'AY'], 'WHY': ['W', 'I'], 
    'WOO': ['W', 'OO'], 'DIME': ['D', 'I', 'M'], 'MAKE': ['M', 'AY', 'K'], 'SEEK': ['S', 'EE', 'K'], 
    'SIDE': ['S', 'I', 'D'], 'WAKE': ['W', 'AY', 'K'], 'WEED': ['W', 'EE', 'D'], 'RAY': ['R', 'AY'], 
    'SHE': ['SH', 'EH'], 'SHY': ['SH', 'I'], 'SHOE': ['SH', 'UU'], 'US': ['UH', 'S'], 
    'COME': ['K', 'UH', 'M'], 'ROCK': ['R', 'AW', 'K'], 'RUM': ['R', 'UH', 'M'], 
    'SHAVE': ['SH', 'EY', 'V'], 'SHOCK': ['SH', 'AA', 'K'], 'VASE': ['V', 'EY', 'S'], 
    'WASH': ['W', 'AW', 'SH'], 'LOW': ['L', 'OE'], 'OATH': ['OE', 'TH'], 'ROW': ['R', 'OE'], 
    'SHOW': ['SH', 'OE'], 'SO': ['S', 'OE'], 'THE': ['DH', 'UH'], 'THOUGH': ['DH', 'OE'], 
    'BASE': ['B', 'AY', 'S'], 'DOME': ['D', 'OE', 'M'], 'LIKE': ['L', 'I', 'K'], 
    'THUMB': ['TH', 'UH', 'M'], 'WILL': ['W', 'IH', 'L'], 'WISH': ['W', 'IH', 'SH'], 
    'CHOW': ['CH', 'OW'], 'COW': ['K', 'OW'], 'HOW': ['H', 'OW'], 'VOW': ['V', 'OW'], 
    'WOW': ['W', 'OW'], 'CHEESE': ['CH', 'EE', 'Z'], 'CHOOSE': ['CH', 'OO', 'Z'], 
    'HATCH': ['H', 'AE', 'CH'], 'HIM': ['H', 'IH', 'M'], 'LOUD': ['L', 'OW', 'D'], 
    'MAD': ['M', 'AE', 'D'], 'MAZE': ['M', 'AY', 'Z'], 'JAY': ['J', 'AY'], 'JOY': ['J', 'OY'], 
    'KNEE': ['N', 'EE'], 'NO': ['N', 'OE'], 'NOW': ['N', 'OW'], 'PAY': ['P', 'AY'], 
    'JOIN': ['J', 'OY', 'N'], 'KEEP': ['K', 'EE', 'P'], 'NOISE': ['N', 'OY', 'Z'], 
    'PEN': ['P', 'EH', 'N'], 'THEM': ['DH', 'EH', 'M'], 'THEN': ['DH', 'EH', 'N'], 
    'GAY': ['G', 'AY'], 'GO': ['G', 'OE'], 'GUY': ['G', 'AY'], 'TIE': ['T', 'AY'], 
    'TOE': ['T', 'OE'], 'TOO': ['T', 'UU'], 'TOY': ['T', 'OY'], 'AZURE': ['AE', 'ZH', 'ER'], 
    'BOOK': ['B', 'UU', 'K'], 'GUN': ['G', 'UH', 'N'], 'PUT': ['P', 'UU', 'T'], 
    'SHIRT': ['SH', 'ER', 'T'], 'SHOULD': ['SH', 'UH', 'D'], 'ALL': ['AW', 'L'], 
    'FEE': ['F', 'EE'], 'OFF': ['AW', 'F'], 'ON': ['AW', 'N'], 'OUGHT': ['AW', 'T'], 
    'YOU': ['Y', 'UU'], 'FOUGHT': ['F', 'AW', 'T'], 'PAWN': ['P', 'AW', 'N'], 
    'RING': ['R', 'IH', 'NG'], 'THING': ['TH', 'IH', 'NG'], 'YOUNG': ['Y', 'UH', 'NG'], 
    'YOUR': ['Y', 'ER']
}

# Phoneme properties
phoneme_to_freq = {'B': 100, 'CH': 400, 'D': 100, 'F': 400, 'G': 100, 'H': 400, 'J': 400, 'K': 100, 'L': 400, 'M': 400, 'N': 400, 'P': 100, 'R': 400, 'S': 400, 'T': 100, 'V': 400, 'W': 400, 'Y': 400, 'Z': 400, 'TH': 300, 'DH': 300, 'SH': 300, 'ZH': 300, 'NG': 60, 'EE': 300, 'AH': 60, 'OO': 300, 'AE': 300, 'AW': 300, 'ER': 300, 'UH': 300, 'IH': 300, 'EH': 300, 'UU': 300, 'AY': 300, 'I': 300, 'OW': 300, 'OE': 300, 'OY': 300}
phoneme_to_modulation = {'B': 30, 'CH': 0, 'D': 30, 'F': 0, 'G': 30, 'H': 0, 'J': 8, 'K': 0, 'L': 30, 'M': 8, 'N': 8, 'P': 0, 'R': 30, 'S': 0, 'T': 0, 'V': 8, 'W': 8, 'Y': 0, 'Z': 8, 'TH': 0, 'DH': 8, 'SH': 0, 'ZH': 8, 'NG': 8, 'EE': 0, 'AH': 0, 'OO': 30, 'AE': 0, 'AW': 0, 'ER': 0, 'UH': 0, 'IH': 0, 'EH': 0, 'UU': 30, 'AY': 30, 'I': 30, 'OW': 0, 'OE': 0, 'OY': 0}
phoneme_to_duration = {'B': 300, 'CH': 300, 'D': 300, 'F': 300, 'G': 300, 'H': 60, 'J': 300, 'K': 300, 'L': 300, 'M': 60, 'N': 60, 'P': 300, 'R': 300, 'S': 300, 'T': 300, 'V': 300, 'W': 60, 'Y': 60, 'Z': 300, 'TH': 400, 'DH': 400, 'SH': 400, 'ZH': 400, 'NG': 400, 'EE': 480, 'AH': 480, 'OO': 480, 'AE': 480, 'AW': 480, 'ER': 480, 'UH': 240, 'IH': 240, 'EH': 240, 'UU': 240, 'AY': 480, 'I': 480, 'OW': 480, 'OE': 480, 'OY': 480}
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
    'NG': [1, 2, 7, 8],
    'EE': {'channels': [6, 5, 4, 3, 2, 1], 'simultaneous': False},
    'AH': {'channels': [[1, 12], [2, 11], [3, 10], [4, 9], [5, 8], [6, 7]], 'simultaneous': True},
    'OO': {'channels': [[18, 24], [17, 23], [16, 22], [15, 21], [14, 20], [13, 19]], 'simultaneous': True},
    'AE': {'channels': [1, 2, 3, 9, 8, 7, 1, 2, 3, 9, 8, 7], 'simultaneous': False},
    'AW': {'channels': [22, 23, 24, 22, 23, 24], 'simultaneous': False},
    'ER': {'channels': [19, 20, 21, 19, 20, 21], 'simultaneous': False},
    'UH': {'channels': [6, 12, 18, 24], 'simultaneous': True},
    'IH': {'channels': [1, 2, 3, 4], 'simultaneous': False},
    'EH': {'channels': [4, 10, 16, 22], 'simultaneous': True},
    'UU': {'channels': [[13, 19],[14, 20], [15, 21], [16, 22]], 'simultaneous': True},
    'AY': {'channels': [[3, 12],[4, 11], [5, 10], [6, 9]], 'simultaneous': True},
    'I': {'channels': [ [13, 19],[14, 20],  [15, 21], [16, 22],[4, 10], [3, 9], [2, 8], [1, 7]], 'simultaneous': True},
    'OW': {'channels': [6, 4, 2], 'simultaneous': False},
    'OE': {'channels': [4, 10, 16, 22, 4], 'simultaneous': False},
    'OY': {'channels': [14, 16, 18], 'simultaneous': False}
}

# Generate waveform
def generate_waveform(phoneme):
    freq, modulation, duration, channels, simultaneous = get_phoneme_properties(phoneme)
    sample_rate = 44100
    t = np.linspace(0, duration / 1000, int(sample_rate * duration / 1000), False)
    tone = np.sin(2 * np.pi * freq * t)
    if modulation:
        modulator = np.sin(2 * np.pi * modulation * t)
        tone = tone * modulator
    return apply_window(tone, duration)

def apply_window(tone, duration, sample_rate=44100):
    window = hanning(int(sample_rate * duration / 1000))
    return tone * window

def get_phoneme_properties(phoneme):
    freq = phoneme_to_freq.get(phoneme, 300)
    modulation = phoneme_to_modulation.get(phoneme, 0)
    duration = phoneme_to_duration.get(phoneme, 300)
    channels_info = phoneme_to_channels.get(phoneme, {'channels': [1, 2, 3, 4], 'simultaneous': False})

    if isinstance(channels_info, dict):
        channels = channels_info.get('channels', [1, 2, 3, 4])
        simultaneous = channels_info.get('simultaneous', False)
    else:
        channels = channels_info
        simultaneous = False

    return freq, modulation, duration, channels, simultaneous

def text_to_phonemes(text):
    words = text.split()
    phonemes = []
    for word in words:
        if word.upper() in phoneme_map:
            phonemes.extend(phoneme_map[word.upper()])
    return phonemes

def play_wave_on_laptop(wave, p, samp_freq):
    stream = p.open(format=pyaudio.paFloat32, channels=2, rate=samp_freq, output=True)
    stream.write(wave.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
def play_wave_on_motu(waveforms, phonemes, device_index, p, samp_freq, simultaneous=False):
    print(f"Playing phonemes: {phonemes}")  # Debug print
    stream = p.open(format=pyaudio.paFloat32, channels=24, rate=samp_freq, output=True, output_device_index=device_index)
    for waveform, phoneme in zip(waveforms, phonemes):
        print(f"Processing phoneme: {phoneme}")  # Debug print
        freq, modulation, duration, channels, simultaneous = get_phoneme_properties(phoneme)
        buffer = np.zeros((len(waveform), 24))
        
        channels_info = phoneme_to_channels.get(phoneme, {'channels': [1, 2, 3, 4], 'simultaneous': False})
        
        if isinstance(channels_info, dict):
            channels = channels_info['channels']
            simultaneous = channels_info.get('simultaneous', False)
        else:
            channels = channels_info
            simultaneous = False

        if simultaneous:
            if isinstance(channels[0], list):  # Check if channels are pairs
                for pair in channels:
                    for ch in pair:
                        buffer[:, int(ch) - 1] = waveform
                    stream.write(buffer.astype(np.float32).tobytes())
                    for ch in pair:
                        buffer[:, int(ch) - 1] = 0
            else:
                for ch_num in channels:
                    buffer[:, int(ch_num) - 1] = waveform
                stream.write(buffer.astype(np.float32).tobytes())
                for ch_num in channels:
                    buffer[:, int(ch_num) - 1] = 0
        else:
            if isinstance(channels, int):  # Check if channel is an integer
                buffer[:, int(channels) - 1] = waveform
            elif isinstance(channels, list):
                for ch_num in channels:
                    buffer[:, int(ch_num) - 1] = waveform
            stream.write(buffer.astype(np.float32).tobytes())
            buffer.fill(0)  # Clear the buffer after playing

        # Add a 1-second delay after each phoneme
        time.sleep(1)

    stream.stop_stream()
    stream.close()


def get_device_index(p):
    device_index = None
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if 'MOTU' in dev['name'] and dev['maxOutputChannels'] == 24:
            device_index = i
            break
    return device_index
def main():
    print("Starting main function")
    p = pyaudio.PyAudio()
    samp_freq = 44100
    print("Initializing PyAudio")
    
    device_index = get_device_index(p)
    print(f"Got device index: {device_index}")
    
    if device_index is None:
        print("MOTU device not found")
        p.terminate()
        return
    
    print(f"MOTU device found at index: {device_index}")
    
    while True:
        input_word = input("Enter a word: ").strip().upper()
        print(f"Received input: {input_word}")
        
        if input_word == 'EXIT':
            break
        if input_word in phoneme_map:
            phonemes = phoneme_map[input_word]
            print(f"Phonemes for {input_word}: {phonemes}")
            
            waveforms = []
            print("Starting waveform generation")
            for phoneme in phonemes:
                try:
                    print(f"Generating waveform for {phoneme}")
                    waveform = generate_waveform(phoneme)
                    waveforms.append(waveform)
                    print(f"Generated waveform for phoneme: {phoneme}")
                except Exception as e:
                    print(f"Error generating waveform for phoneme {phoneme}: {str(e)}")
            
            print(f"Generated {len(waveforms)} waveforms")
            
            if waveforms:
                try:
                    print("Starting play_wave_on_motu")
                    play_wave_on_motu(waveforms, phonemes, device_index, p, samp_freq, simultaneous=False)
                    print("Finished play_wave_on_motu")
                except Exception as e:
                    print(f"Error in play_wave_on_motu: {str(e)}")
            else:
                print("No waveforms generated")
        else:
            print("Word not found in the list.")

    print("Exiting main loop")
    p.terminate()
    print("PyAudio terminated")

if __name__ == "__main__":
    print("Script starting")
    main()
    print("Script finished")