from Training import Training
from get_params import get_params

Training(
    get_params(),
    cache_dir = './arxiv',
    categories = [
        'Machine Learning',
        'Neural and Evolutionary Computing',
        'Optimization'
    ],
    keywords = [
        'neural',
        'network',
        'deep'
    ]
    )()