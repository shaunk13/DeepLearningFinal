import tensorflow as tf

def checkerboard(height, width, r=False):
    """
    Takes in the dimensions of the checkerboard mask to be produced.
    Will output a 2D tf tensor containing a checkerboard pattern of 1s and 0s.
    """
    checkerboard = [[(i + j) % 2 for j in range(width)] for i in range(height)]
    return tf.convert_to_tensor(checkerboard)