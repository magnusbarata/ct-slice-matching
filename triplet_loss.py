import tensorflow as tf

def paired_chi_squared(X, Y, eps=1e-16):
    """TODO
    """
    return tf.reduce_sum((X - Y)**2 / tf.maximum(X + Y, eps), axis=1)

def paired_manhattan(X, Y):
    """TODO
    """
    return tf.reduce_sum(tf.abs(X - Y), axis=1)

def paired_cosine(X, Y):
    """Compute paired cosine distances between X and Y, elementwise.

    # Args
        X: (n_vectors, n_features) shaped tensor or numpy array
        Y: (n_vectors, n_features) shaped tensor or numpy array
        squared: bool, output is squared euclidean distance if true.

    # Returns
        (n_vectors,) shaped paired distances of X and Y
    """
    X /= tf.sqrt(squared_norm(X))
    Y /= tf.sqrt(squared_norm(Y))
    return .5 * squared_norm(X - Y, False)

def squared_norm(X, keepdims=True):
    return tf.reduce_sum(X*X, axis=1, keepdims=keepdims)

def euclidean_distance_mat(X, Y=None, squared=False):
    """Compute euclidean distance matrix between each pair of vectors.

    # Args
        X: tensor or numpy array with shape of (n_vectors_a, n_features)
        Y: tensor or numpy array with shape of (n_vectors_b, n_features)
        squared: bool, output is squared euclidean distance if true.

    # Returns
        2D euclidean distance matrix of shape (n_vectors_a, n_vectors_b)
    """
    if Y is None: Y = X

    XX = tf.reduce_sum(X*X, axis=1, keepdims=True)
    YY = tf.reduce_sum(Y*Y, axis=1, keepdims=True)
    dist = XX - 2.0 * tf.matmul(X, Y, transpose_b=True) + tf.transpose(YY)
    dist = tf.maximum(dist, 0.0) # preventing negative distance

    # due to float rounding errors we need to ensure that distances between vectors and themselves are set to 0.0
    if X is Y: dist *= (1.0 - tf.eye(tf.shape(dist)[0]))

    if not squared:
        mask = tf.to_float(tf.equal(dist, 0.0)) # mask to help compute sqrt on 0.0 elements
        dist = dist + mask * 1e-16

        dist = tf.sqrt(dist)
        dist *= (1.0 - mask) # reverting the mask

    return dist

def paired_euclidean(X, Y, squared=False):
    """Compute paired distances between X and Y, elementwise.

    # Args
        X: (n_vectors, n_features) shaped tensor or numpy array
        Y: (n_vectors, n_features) shaped tensor or numpy array
        squared: bool, output is squared euclidean distance if true.

    # Returns
        (n_vectors,) shaped paired distances of X and Y
    """
    distances = tf.reduce_sum(tf.square(X - Y), axis=1)
    return distances if squared else tf.sqrt(distances)

PAIRED_DISTANCES = {'euclidean': paired_euclidean,
                    'manhattan': paired_manhattan,
                    'cosine': paired_cosine,
                    'chi_squared': paired_chi_squared}

def get_triplet_loss(margin, distance_function, use_slice_dist=False):
    """Function closure returning triplet loss function.

    # Args
        margin: margin value between positive and negative samples.
        distance_function: function to compute paired distance between samples.

    # Returns
        triplet loss function
    """
    def triplet_loss(y_true, y_pred):
        anchor, neg, pos = tf.split(y_pred, num_or_size_splits=3, axis=0)
        n_dist = distance_function(anchor, neg)
        p_dist = distance_function(anchor, pos)

        y_a, y_n, _ = tf.split(tf.squeeze(y_true, axis=-1), num_or_size_splits=3)
        slice_dist = tf.abs(y_a - y_n) * 0.05 if use_slice_dist else 0.0
        loss = tf.maximum(p_dist - n_dist + margin + slice_dist, 0.0)
        return tf.reduce_mean(loss, axis=0)

    return triplet_loss
