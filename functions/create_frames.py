import numpy as np


def create_frames(input_vector, hop, window_size):
    number_slices = np.floor((len(input_vector) - window_size) / hop).astype(int)
    input_vector = input_vector[:number_slices * hop + window_size]

    vector_frames = np.zeros(((len(input_vector) - 1) // hop, window_size))
    for index in range(number_slices + 1):
        index_time_start = index * hop
        index_time_end = index * hop + window_size

        vector_frames[index] = input_vector[index_time_start: index_time_end]

    return vector_frames, number_slices
