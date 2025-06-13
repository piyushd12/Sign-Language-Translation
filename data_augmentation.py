import numpy as np

def augment_keypoints(keypoints, noise_level=0.01):
    """
    Augment keypoints by adding random noise.

    Args:
        keypoints (numpy.ndarray): Original keypoints of shape (10, 126).
        noise_level (float): Standard deviation of the noise to be added.

    Returns:
        numpy.ndarray: Augmented keypoints.
    """
    noise = np.random.normal(0, noise_level, keypoints.shape)
    return keypoints + noise

def temporal_augmentation(keypoints, drop_rate=0.1):
    """
    Perform temporal augmentation by randomly dropping frames.

    Args:
        keypoints (numpy.ndarray): Original keypoints of shape (10, 126).
        drop_rate (float): Fraction of frames to drop.

    Returns:
        numpy.ndarray: Augmented keypoints with dropped frames.
    """
    num_frames = keypoints.shape[0]
    keep_indices = np.random.choice(num_frames, int(num_frames * (1 - drop_rate)), replace=False)
    keep_indices = np.sort(keep_indices)
    return keypoints[keep_indices]
