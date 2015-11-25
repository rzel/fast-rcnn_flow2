# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

# import datasets.pascal_voc
import datasets.imagenet_vid
import numpy as np

"""def _selective_search_IJCV_top_k(split, year, top_k):
    imdb = datasets.pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb
"""

year = '2015'

# Set up voc_<year>_<split> using selective search "fast" mode
for split in ['train']: #, 'val', 'trainval', 'test']:
    name = 'vid_{}'.format( split)
    __sets[name] = (lambda split=split, year=year:
            datasets.imagenet_vid(split, year))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
