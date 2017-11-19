import math


def euclidean_distance(vector1, vector2):
    points = zip(vector1, vector2)
    diffs_squared_distance = [pow((a - b), 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))


def cosin_distance(vector1, vector2):
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for (a, b) in zip(vector1, vector2):
        dot_product += a * b
        norm_a += a ** 2
        norm_b += b ** 2
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot_product / ((norm_a * norm_b) ** 0.5)


def cosin_distance(vector1, vector2):
    from sklearn.metrics.pairwise import cosine_distances
    return cosine_distances(vector1, vector2)


def cosine_similarity(vector1, vector2):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(vector1, vector2)
