if [ $# -ne 3 ]; then
    echo "Usage: ./run.sh <input.wav> <output.wav> <time_stretch_ratio>"
    exit 1
fi

# Входные данные
input_file=$1
output_file=$2
time_stretch_ratio=$3

# Стандартные переменные
window_size=1024
hop_size=256
step=$(echo "12 * l($time_stretch_ratio) / l(2)" | bc -l)

python - << END
from scipy.io.wavfile import read, write
from functions import pitch_shift
import numpy as np

sample_rate, data = read("$input_file")
out = pitch_shift(data, $window_size, $hop_size, $step)
write("$output_file", int(sample_rate // $time_stretch_ratio), out)
END
