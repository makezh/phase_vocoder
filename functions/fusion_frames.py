import numpy as np


def fusion_frames(frames_matrix, hop):
    number_frames, size_frames = frames_matrix.shape
    vector_time = np.zeros(number_frames * hop - hop + size_frames)
    time_index = 0

    for index in range(number_frames):
        vector_time[time_index:time_index + size_frames] += frames_matrix[index, :]
        time_index += hop

    return vector_time
