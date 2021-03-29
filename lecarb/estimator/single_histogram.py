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
    def __init__(self, state, num_bins, table):
        super(SHist, self).__init__(table=table, bins=num_bins)
        self.bins = state['partitions']
        self.total = state['total']
        self.encoder_map = state['encoder']
        self.categorical_variables = state['categorical_variables']


    def query(self, query):
        #print('in query')
        histograms = self.bins
        categorical_variables = self.categorical_variables
        encoder_map = self.encoder_map
        total = self.total
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        # print(columns)
        # print(operators)
        # print(values)
        est_card = []
        #Step 1: Encode categorical to numerical from query statement
        for i in range (0, len(columns)):
            if columns[i] in categorical_variables:
                encoder = encoder_map[columns[i]]
                val = []
                val.append(values[i])
                ans = encoder.transform(val)
                values[i] = ans[0]
                val = []
        # print('Encoded values:')
        # print(values)
        
        start_stmp = time.time()
        #Iterate over all conditions
        for i in range(0, len(columns)):
            #Check for specific condition corresponding to the operator
            #print(histograms)
            bins = histograms[columns[i]]
            #print('in loop')
            if operators[i] == '=':
                counter = 0
                for k in range(0,len(bins)):
                    bin = bins[k]
                    # if columns[i] == 'education':
                    #     print(values[i])
                    if values[i] >= bin[0] and values[i] <= bin[1]:
                        counter = counter + bin[3]/bin[4]

                est_card.append(counter)

            elif operators[i] == '<=' or operators[i] == '<':
                if operators[i] == '<=':
                    est_card.append(self.handle_less_thanEqual_case(bins, values[i]))
                #print('less than case')
                else:
                    est_card.append(self.handle_less_than_case(bins, values[i]))
            
            elif operators[i] == '>=' or operators[i] == '>':
                #print('greater than case')
                if operators[i] == '>=':
                    est_card.append(self.handle_greater_thanEqual_case(bins, values[i]))
                else:
                    est_card.append(self.handle_greater_than_case(bins, values[i]))
            
            else:
                est_card.append(self.handle_inBetween_case(bins, values[i]))
        
        for i in range(0, len(est_card)):
            #print('before: ', est_card[i])
            est_card[i] = est_card[i]/total
            #print('after: ', est_card[i])
        #print(np.prod(est_card))
        dur_ms = (time.time() - start_stmp) * 1e3
            
        return np.prod(est_card)*total, dur_ms

    def handle_inBetween_case(self, bins, value):
        lower_bound = value[0]
        upper_bound = value[1]
        counter = 0
        for k in range(0,len(bins)):
            bin = bins[k]
            if bin[0] >= lower_bound:
                if bin[1] <= upper_bound:
                    counter = counter + bin[3]
                elif bin[1] > upper_bound and upper_bound > bin[0]:
                    counter = counter + (bin[3]*(upper_bound-bin[0]))/(bin[1]-bin[0]) 
        return counter

    def handle_less_than_case(self, bins, value):
        counter = 0
        for k in range(0, len(bins)):
            bin = bins[k]
            if value > bin[1] and value > bin[0]:
                counter = counter + bin[3]
            elif value > bin[0] and value < bin[1]:
                counter = counter + (bin[3]*(value-bin[0]))/(bin[1]-bin[0]) 
        #print('less than')
        #print(counter)
        return counter
    
    def handle_less_thanEqual_case(self, bins, value):
        counter = 0
        for k in range(0, len(bins)):
            bin = bins[k]
            if value >= bin[1] and value >= bin[0]:
                counter = counter + bin[3]
            elif value >= bin[0] and value < bin[1]:
                counter = counter + (bin[3]*(value-bin[0]))/(bin[1]-bin[0]) 
        #print('less than')
        #print(counter)
        return counter
    
    
    def handle_greater_than_case(self, bins, value):
        counter = 0
        for k in range(0, len(bins)):
            bin = bins[k]
            if bin[1] > value and bin[0] > value:
                counter = counter + bin[3]
            elif value > bin[0] and value < bin[1]:
                counter = counter + bin[3]*((bin[1]-value))/(bin[1]-bin[0])
        #print('greater than')
        #print(counter)
        return counter
    
    def handle_greater_thanEqual_case(self, bins, value):
        counter = 0
        for k in range(0, len(bins)):
            bin = bins[k]
            if bin[1] >= value and bin[0] >= value:
                counter = counter + bin[3]
            elif value <= bin[1] and bin[0] <= value:
                counter = counter + bin[3]*((bin[1]-value))/(bin[1]-bin[0])
        #print('greater than')
        #print(counter)
        return counter
        
def construct_bins(table, num_bins):
    partitions = {}
    data = table.data
    ## Converting categorical values to numerical
    encoder_map = {}
    categorical_variables = []
    for i in range(0, len(data.dtypes)):
        if data.dtypes[i].name == 'category':
            categorical_variables.append(data.columns[i])
            encoder = LabelEncoder()
            data[data.columns[i]] = encoder.fit_transform(data[data.columns[i]])
            encoder_map[data.columns[i]] = encoder
    #print(encoder_map)
    data_sorted = data.copy()
    for col in data_sorted:
        data_sorted[col] = data_sorted[col].sort_values(ignore_index=True)
    data_sorted_numpy = data_sorted.to_numpy()
    #print(data_sorted_numpy)
    average_freq = len(data_sorted_numpy)/num_bins
    total = len(data_sorted_numpy)
    
    #Create 1-D hist for each attribute
    for i in range(0,len(data_sorted.columns)):
        k = 1
        #print('FOR: ',data_sorted.columns[i])
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
            #print(meta)
            bins.append(meta)
        partitions[data_sorted.columns[i]] = bins
        #partitions.append(bins)
        state = {'partitions':partitions, 'total': total, 'encoder': encoder_map, 'categorical_variables': categorical_variables}
    
    return state

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
    model_file = model_path / f"{table.version}-shist_bin{params['num_bins']}.pkl"

    if model_file.is_file():
        L.info(f"{model_file} already exists, directly load and use")
        with open(model_file, 'rb') as f:
            state = pickle.load(f)
    else:
        L.info(f"Construct SHist with at most {params['num_bins']} bins...")
        state = construct_bins(table, params['num_bins'])
        with open(model_file, 'wb') as f:
            pickle.dump(state, f, protocol=PKL_PROTO)
        L.info(f"MHist saved to {model_file}")

    # partitions = attribute_bins
    estimator = SHist(state, params['num_bins'], table)
    L.info(f"Built SHist estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite)
