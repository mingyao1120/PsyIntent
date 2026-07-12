from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(osp.abspath(__file__))
# Add reproduction/lib and reproduction/src
parent_dir = osp.dirname(this_dir)
add_path(osp.join(parent_dir, 'lib'))
add_path(parent_dir)
