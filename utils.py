import numpy as np

def dot_product(v1, v2):
    '''
    v1 and v2 are vectors of same shape.
    return the scalar dot product of the two vectors.
    '''
    return np.dot(v1, v2)

def cosine_similarity(v1, v2):
    '''
    v1 and v2 are vectors of same shape.
    Return the cosine similarity between the two vectors.
    
    The formula for cosine similarity is: 
    (v1 dot v2) / (||v1|| * ||v2||)
    '''
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product(v1, v2) / (norm_v1 * norm_v2)

def nearest_neighbor(target_vector, vectors):
    '''
    target_vector is a vector of shape d.
    vectors is a matrix of shape N x d.
    return the row index of the vector in vectors that is closest to 
    target_vector in terms of cosine similarity.
    '''
    max_similarity = -1  # Initialize with the lowest possible similarity
    nearest_index = -1    # To store the index of the closest vector
    
    for i, vector in enumerate(vectors):
        similarity = cosine_similarity(target_vector, vector)
        if similarity > max_similarity:
            max_similarity = similarity
            nearest_index = i
            
    return nearest_index
