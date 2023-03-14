import numpy as np

from functions import fusion_frames, create_frames


def pitch_shift(input_vector: np.ndarray,
                window_size: int,
                hop_size: int,
                step: int,
                max_amplitude: float = 0.9):
    alpha = 2 ** (step / 12)
    hop_out = round(alpha * hop_size)
    is_stereo = False
    x = input_vector

    if x.ndim == 2:
        is_stereo = True
        x = (x[:, 0] + x[:, 1]) / 2

    x = np.concatenate((np.zeros(hop_size * 3), x))

    wn = np.hanning(window_size * 2 + 1)[1::2]

    y, number_frames_input = create_frames(x, hop_size, window_size)

    number_frames_output = number_frames_input
    output_y = np.zeros((number_frames_output, window_size))

    phase_cumulative = 0
    previous_phase = 0

    for index in range(number_frames_input):
        # Анализ
        current_frame = y[index, :]
        current_frame_windowed = current_frame * wn / np.sqrt((window_size / hop_size) / 2)
        current_frame_windowed_fft = np.fft.fft(current_frame_windowed)
        mag_frame = np.abs(current_frame_windowed_fft)
        phase_frame = np.angle(current_frame_windowed_fft)

        # Обработка
        delta_phi = phase_frame - previous_phase
        previous_phase = phase_frame
        delta_phi_prime = delta_phi - 2 * hop_size * np.pi * np.arange(window_size) / window_size
        delta_phi_prime_mod = np.mod(delta_phi_prime + np.pi, 2 * np.pi) - np.pi
        true_freq = 2 * np.pi * np.arange(window_size) / window_size + delta_phi_prime_mod / hop_size
        phase_cumulative += hop_out * true_freq

        # Синтез
        output_mag = mag_frame
        output_frame = np.real(np.fft.ifft(output_mag * np.exp(1j * phase_cumulative)))
        output_y[index, :] = output_frame * wn / np.sqrt((window_size / hop_out) / 2)

    output_time_stretched = fusion_frames(output_y, hop_out)

    output_time = np.interp(
        np.arange(0, len(output_time_stretched), alpha),
        np.arange(len(output_time_stretched)),
        output_time_stretched,
        left=0, right=0
    )

    # Нормализация
    output_time = output_time / max(abs(output_time_stretched)) * max_amplitude

    if is_stereo:
        result = np.zeros((len(output_time), 2))
        result[:, 0] = result[:, 1] = output_time
        return result

    return output_time
