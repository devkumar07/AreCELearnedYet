import enum
import copy
import logging
import pickle
import time
import threading
import bisect
from typing import Any, Dict, Tuple
import numpy as np

from .estimator import Estimator
from .utils import run_test
from ..constants import MODEL_ROOT, NUM_THREADS, PKL_PROTO
from ..dtypes import is_categorical
from ..dataset.dataset import load_table
from ..workload.workload import query_2_triple
from sklearn.preprocessing import LabelEncoder

L = logging.getLogger(__name__)

class SHist(Estimator):
    def __init__(self, bins, encoder_map, table):
        super(SHist, self).__init__(table=table, bins=len(bins))
        self.bins = bins
        self.encoder_map = encoder_map

    def query(self, query):
        #print('in query')
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        print(columns)
        print(operators)
        print(values)
        est_card = []
        for pid in candidate_pids:
            est_card.append(self.partitions[pid].query(columns, operators, values))

        dur_ms = (time.time() - start_stmp) * 1e3

        #  return np.round(est.card.sum()), dur_ms
        return np.round(np.sum(est_card)), dur_ms
        
def construct_bins(table, num_bins):
    partitions = []
    data = table.data
    ## Converting categorical values to numerical
    encoder_map = {}
    for i in range(0, len(data.dtypes)):
        if data.dtypes[i].name == 'category':
            encoder = LabelEncoder()
            data[data.columns[i]] = encoder.fit_transform(data[data.columns[i]])
            encoder_map[data.columns[i]] = encoder
    print(encoder_map)
    #data['workclass'] = encoder_map['workclass'].inverse_transform(data['workclass'])
    #print(data)
    #data_sorted = data.sort_values(by=list(data.columns),axis=0, ascending = True)
    data_sorted = data.copy()
    for col in data_sorted:
        data_sorted[col] = data_sorted[col].sort_values(ignore_index=True)
    data_sorted_numpy = data_sorted.to_numpy()
    print(data_sorted_numpy)
    average_freq = len(data_sorted_numpy)/num_bins
    
    #Create 1-D hist for each attribute
    for i in range(0,len(data_sorted.columns)):
        k = 1
        print('FOR: ',data_sorted.columns[i])
        #Iterate over num_bins
        bins = []
        for x in range(num_bins): 
            start = data_sorted_numpy[k][i]
            freq = 1
            values = []
            values.append(data_sorted_numpy[k][i])
            while k % int(average_freq) != 0:
                values.append(data_sorted_numpy[k][i])
                k = k + 1
                freq = freq + 1
            values.append(data_sorted_numpy[k][i])
            finish = data_sorted_numpy[k][i]
            k = k + 1
            freq = freq + 1
            distinct_val = np.unique(values)
            meta = [start,finish,'True',freq,len(distinct_val)] 
            print(meta)
            bins.append(meta)
        partitions.append(bins)
    
    return partitions, encoder_map

def test_single_hist(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        version: the version of table that the histogram is built from, might not be the same with the one we test on
        num_bins: maximum number of partitions
    """
    # prioriy: params['version'] (draw sample from another dataset) > version (draw and test on the same dataset)
    table = load_table(dataset, params.get('version') or version)

    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}-mhist_bin{params['num_bins']}.pkl"

    if model_file.is_file():
        L.info(f"{model_file} already exists, directly load and use")
        with open(model_file, 'rb') as f:
            state = pickle.load(f)
    else:
        L.info(f"Construct SHist with at most {params['num_bins']} bins...")
        state, encoder = construct_bins(table, params['num_bins'])
        with open(model_file, 'wb') as f:
            pickle.dump(state, f, protocol=PKL_PROTO)
        L.info(f"MHist saved to {model_file}")

    # partitions = attribute_bins
    estimator = SHist(state, encoder, table)
    L.info(f"Built SHist estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite)