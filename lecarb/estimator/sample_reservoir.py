
import time
import logging
from typing import Any, Dict
import numpy as np
import pandas as pd
from .estimator import Estimator, OPS
from .utils import run_test
from ..workload.workload import query_2_triple
from ..dataset.dataset import load_table
import random
from random import randrange
import math
import time
from pandas.util.testing import assert_frame_equal

L = logging.getLogger(__name__)

class Sampling(Estimator):
    def __init__(self, table, ratio, seed, size):
        super(Sampling, self).__init__(table=table, version=table.version, ratio=ratio, seed=seed)
        k = size#int(input('Reservoir size: '))
        #k = 10000 #reservoir size
        n = len(table.data) #dataset size
        data = table.data
        data_vec = data.to_numpy()

        #Initialize reservoir
        start_time = time.time()
        Sample = pd.DataFrame(data = data.iloc[0:k],columns=data.columns)
        Sample_vec = Sample.to_numpy()
        
        #Setting value of W
        W = math.exp(math.log10(random.random()/k))
        
        #print(Sample)
        
        S = k+1
        
        while S <= n:
            S = S + math.floor(math.log10(random.random())/math.log10(1-W)) + 1
            if S <= n:
                Sample_vec[randrange(k)] = data_vec[S]
                W = W * math.exp(math.log10(random.random())/k)
        result = pd.DataFrame(Sample_vec, columns = data.columns)
        self.sample = result
        print('EXECUTION TIME: ',time.time() - start_time)
        #print('AFTER------------------')
        #print(self.sample)
        #print(Sample.equals(Sample_vec))
        self.sample_num = len(self.sample)

    def query(self, query):
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        start_stmp = time.time()
        bitmap = np.ones(self.sample_num, dtype=bool)
        for c, o, v in zip(columns, operators, values):
            bitmap &= OPS[o](self.sample[c], v)
        card = np.round((self.table.row_num / self.sample_num) * bitmap.sum())
        dur_ms = (time.time() - start_stmp) * 1e3
        return card, dur_ms

def test_sample_reservoir(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        version: the version of table that the sample draw from, might not be the same with the one we test on
        ratio: the ratio of the sample
    """
    # prioriy: params['version'] (draw sample from another dataset) > version (draw and test on the same dataset)
    table = load_table(dataset, params.get('version') or version)
    s = int(input('Reservoir size: '))
    L.info("construct reservoir sampling estimator...")
    estimator = Sampling(table, ratio=params['ratio'] or 0.01, seed=seed, size = s)
    L.info(f"built reservoir sampling estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite)


