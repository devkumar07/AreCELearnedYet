
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

L = logging.getLogger(__name__)

class Sampling(Estimator):
    def __init__(self, table, ratio, seed):
        super(Sampling, self).__init__(table=table, version=table.version, ratio=ratio, seed=seed)
        k = int(input('Reservoir size: '))
        #k = 10000 #reservoir size
        n = len(table.data) #dataset size
        data = table.data

        #Initialize reservoir
        Sample = pd.DataFrame(data = data.iloc[0:k],columns=data.columns)
        
        #Setting value of W
        W = math.exp(math.log(random.random()/k))
        
        #print(Sample)
        
        S = k+1
        
        while S <= n:
            S = S + math.floor(math.log(random.random())/math.log(1-W)) + 1
            if S <= n:
                Sample.loc[randrange(k)] = data.iloc[S]  
                W = W * math.exp(math.log(random.random())/k)

        self.sample = Sample
        #print('AFTER------------------')
        #print(self.sample)
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
    L.info("construct reservoir sampling estimator...")
    estimator = Sampling(table, ratio=params['ratio'] or 0.01, seed=seed)
    L.info(f"built reservoir sampling estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite)


