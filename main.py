from scipy.io.wavfile import read, write
from functions import pitch_shift
import numpy as np

if __name__ == "__main__":
    input_file = "music/test_mono.wav"
    output_file = "music/test_mono_r05.wav"
    time_stretch_ratio = 0.5
    window_size = 1024
    hop_size = 256
    step = 12 * np.log2(time_stretch_ratio)

    sample_rate, data = read(input_file)
    out = pitch_shift(data, 1024, 256, step)
    write(output_file, int(sample_rate // time_stretch_ratio), out)
