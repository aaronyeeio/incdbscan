import sys
from datetime import datetime
from pathlib import Path

from line_profiler import LineProfiler

from incdbscan import IncrementalDBSCAN
from incdbscan._bfscomponentfinder import BFSComponentFinder
from incdbscan._deleter import Deleter
from incdbscan._inserter import Inserter
from incdbscan._neighbor_searcher import NeighborSearcher
from incdbscan._object import Object
from incdbscan._objects import Objects
from incdbscan.tests.testutils import (
    read_chameleon_data,
    read_handl_data
)


BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'incdbscan' / 'tests' / 'data'


def test1():
    data = read_handl_data()

    algo = IncrementalDBSCAN(eps=1)
    inserted_objects = algo.insert(data)
    # Get IDs of inserted objects for deletion
    inserted_ids = [obj.id for obj in inserted_objects]
    algo.delete(inserted_ids)


def test2():
    data = read_chameleon_data()[:2000]

    algo = IncrementalDBSCAN(eps=10)
    inserted_objects = algo.insert(data)
    # Get IDs of inserted objects for deletion
    inserted_ids = [obj.id for obj in inserted_objects]
    algo.delete(inserted_ids)


def test_soft_clustering():
    data = read_chameleon_data()[:2000]

    algo = IncrementalDBSCAN(eps=10, min_pts=5, eps_soft=20)
    inserted_objects = algo.insert(data)

    # Get IDs of inserted objects for soft clustering
    inserted_ids = [obj.id for obj in inserted_objects]

    # Test soft clustering with different kernels
    algo.get_soft_labels(inserted_ids, kernel='gaussian')
    algo.get_soft_labels(inserted_ids, kernel='inverse')
    algo.get_soft_labels(inserted_ids, kernel='linear')

    # Test without noise probability
    algo.get_soft_labels(inserted_ids, include_noise_prob=False)


def print_profile(test, tag=''):
    profiler = LineProfiler()
    # profiler.add_module(Inserter)
    # profiler.add_module(Deleter)
    profiler.add_module(IncrementalDBSCAN)
    # profiler.add_module(Objects)
    # profiler.add_module(BFSComponentFinder)
    # profiler.add_module(Object)
    # profiler.add_module(NeighborSearcher)

    wrapper = profiler(test)
    wrapper()

    timestamp = str(datetime.now())[:19]
    filename = f'{timestamp}_{test.__name__}{tag}.txt'
    profile_path = BASE_PATH / 'profiling' / filename

    with open(profile_path, 'w') as f:
        profiler.print_stats(stream=f)


if __name__ == "__main__":
    tag = '_' + sys.argv[1] if len(sys.argv) > 1 else ''
    for test in [test_soft_clustering]:
        print(f'{datetime.now()} Creating profile for {test.__name__} ...')
        print_profile(test, tag)
