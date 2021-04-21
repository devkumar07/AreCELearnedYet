import enum
import copy
import logging
import pickle
import time
import threading
import bisect
from typing import Any, Dict, Tuple
import numpy as np
import random
import mmh3

from .estimator import Estimator
from .utils import run_test
from ..constants import MODEL_ROOT, NUM_THREADS, PKL_PROTO
from ..dtypes import is_categorical
from ..dataset.dataset import load_table
from ..workload.workload import query_2_triple
from sklearn.preprocessing import LabelEncoder

L = logging.getLogger(__name__)

class Sketches_Hist(Estimator):
    def __init__(self, state, num_bins, table):
        super(Sketches_Hist, self).__init__(table=table, bins=num_bins)
        self.bins = state['partitions']
        self.total = state['total']
        self.encoder_map = state['encoder']
        self.categorical_variables = state['categorical_variables']
        self.hash_functions = state['hash_functions']
        self.sketches = state['sketches']
        self.num_bins = state['num_bins']


    def query(self, query):
        histograms = self.bins
        categorical_variables = self.categorical_variables
        encoder_map = self.encoder_map
        total = self.total
        sketches = self.sketches
        hash_functions = self.hash_functions
        num_bins = self.num_bins
        columns, operators, values = query_2_triple(query, with_none=False, split_range=False)
        
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
        
        start_stmp = time.time()

        for i in range(0, len(columns)):
            sketches_buckets = []
            if columns[i] in sketches and (operators[i] == '=' or operators[i] == '[]'):
                sketches_buckets = sketches[columns[i]]
                #num_bins = len(sketches_buckets[0])
                if operators[i] == '=':
                    val1 = sketches_buckets[0][compute_hash(values[i],hash_functions[0],hash_functions[0],num_bins)]
                    val2 = sketches_buckets[1][compute_hash(values[i],hash_functions[1],hash_functions[1],num_bins)]
                    val3 = sketches_buckets[2][compute_hash(values[i],hash_functions[2],hash_functions[2],num_bins)]
                    val4 = sketches_buckets[3][compute_hash(values[i],hash_functions[3],hash_functions[3],num_bins)]
                    val5 = sketches_buckets[4][compute_hash(values[i],hash_functions[4],hash_functions[4],num_bins)]
                   
                    est_card.append(min(val1,val2,val3,val4,val5))
                else:
                    minBound = int(np.rint(values[i][0]))
                    maxBound = int(np.rint(values[i][1]))
                    range_sum = []
                    
                    for y in range(0,5):
                        val = 0
                        for x in range(minBound, maxBound+1):
                            val = val + sketches_buckets[y][compute_hash(x,hash_functions[y],hash_functions[y],num_bins)]
                        range_sum.append(val)
    
                    est_card.append(min(range_sum))

            else:
                bins = histograms[columns[i]]
                if operators[i] == '=':
                    counter = 0
                    for k in range(0,len(bins)):
                        bin = bins[k]
                        if values[i] >= bin[0] and values[i] <= bin[1]:
                            counter = counter + bin[3]/bin[4]

                    est_card.append(counter)

                elif operators[i] == '<=' or operators[i] == '<':
                    if operators[i] == '<=':
                        est_card.append(self.handle_less_thanEqual_case(bins, values[i]))
                    else:
                        est_card.append(self.handle_less_than_case(bins, values[i]))
                
                elif operators[i] == '>=' or operators[i] == '>':
                    if operators[i] == '>=':
                        est_card.append(self.handle_greater_thanEqual_case(bins, values[i]))
                    else:
                        est_card.append(self.handle_greater_than_case(bins, values[i]))
                
                else:
                    est_card.append(self.handle_inBetween_case(bins, values[i]))
        
        for i in range(0, len(est_card)):
            est_card[i] = est_card[i]/total

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
        return counter
    
    def handle_less_thanEqual_case(self, bins, value):
        counter = 0
        for k in range(0, len(bins)):
            bin = bins[k]
            if value >= bin[1] and value >= bin[0]:
                counter = counter + bin[3]
            elif value >= bin[0] and value < bin[1]:
                counter = counter + (bin[3]*(value-bin[0]))/(bin[1]-bin[0]) 
        return counter
    
    
    def handle_greater_than_case(self, bins, value):
        counter = 0
        for k in range(0, len(bins)):
            bin = bins[k]
            if bin[1] > value and bin[0] > value:
                counter = counter + bin[3]
            elif value > bin[0] and value < bin[1]:
                counter = counter + bin[3]*((bin[1]-value))/(bin[1]-bin[0])
        return counter
    
    def handle_greater_thanEqual_case(self, bins, value):
        counter = 0
        for k in range(0, len(bins)):
            bin = bins[k]
            if bin[1] >= value and bin[0] >= value:
                counter = counter + bin[3]
            elif value <= bin[1] and bin[0] <= value:
                counter = counter + bin[3]*((bin[1]-value))/(bin[1]-bin[0])
        return counter

def generate_hash_functions():
    seeds = []
    for i in range(0,5):
        seeds.append(random.randint(1,100000))
    
    return seeds

def compute_hash(value, coeff, const, num_bins):

    pos = mmh3.hash(str(value),coeff) % num_bins
    
    return pos

def construct_bins(table, num_bins, hist_bins):
    partitions = {}
    data = table.data
    encoder_map = {}
    categorical_variables = []
    attribute_sketches = {}

    #Step 1: Generate Hash functions
    hash_functions = generate_hash_functions()

    #Step 2: Convert categorical to numerical
    start_time = time.time()
    for i in range(0, len(data.dtypes)):
        if data.dtypes[i].name == 'category':
            categorical_variables.append(data.columns[i])
            encoder = LabelEncoder()
            data[data.columns[i]] = encoder.fit_transform(data[data.columns[i]])
            encoder_map[data.columns[i]] = encoder

    data_sorted = data.copy()


    #Step 3: sort columns
    for col in data_sorted:
        data_sorted[col] = data_sorted[col].sort_values(ignore_index=True)
    
    data_sorted_numpy = data_sorted.to_numpy()
    total = len(data_sorted_numpy)
    
    original_bins = num_bins
    original_hist_bins = hist_bins
    average_freq = int(len(data_sorted_numpy)/hist_bins)
    start_index = 0
    end_index = average_freq

    #Step 4: Construct sketch/histogram for each attribute
    for i in range(0,len(data_sorted.columns)):
        bins = []
        
        bucket1 = [0] *  num_bins
        bucket2 = [0] *  num_bins
        bucket3 = [0] *  num_bins
        bucket4 = [0] *  num_bins
        bucket5 = [0] *  num_bins

        if (data.columns[i] in categorical_variables or len(data[data.columns[i]].unique()) < 100):
            counts = data_sorted[data.columns[i]].value_counts().to_dict()
            for x in counts:
                pos1 = compute_hash(x, hash_functions[0], hash_functions[0], num_bins)
                pos2 = compute_hash(x, hash_functions[1], hash_functions[1], num_bins)
                pos3 = compute_hash(x, hash_functions[2], hash_functions[2], num_bins)
                pos4 = compute_hash(x, hash_functions[3], hash_functions[3], num_bins)
                pos5 = compute_hash(x, hash_functions[4], hash_functions[4], num_bins)

                bucket1[pos1] = bucket1[pos1] + counts[x]
                bucket2[pos2] = bucket2[pos2] + counts[x]
                bucket3[pos3] = bucket3[pos3] + counts[x]
                bucket4[pos4] = bucket4[pos4] + counts[x]
                bucket5[pos5] = bucket5[pos5] + counts[x]
            buckets = [bucket1, bucket2, bucket3, bucket4, bucket5]
            attribute_sketches[data.columns[i]] = buckets

        if data.columns[i] not in categorical_variables or data.columns[i] not in attribute_sketches:
            if len(data[data.columns[i]].unique()) < hist_bins:
                hist_bins = len(data[data.columns[i]].unique())
                counts = data_sorted[data.columns[i]].value_counts().to_dict()
                for key in counts:
                    meta = [key,key,'True',counts[key],1] 
                    bins.append(meta)
            else:
                for x in range(0,hist_bins):
                    values = data_sorted[data.columns[i]][start_index:end_index + 1]
                    start = values[start_index]
                    finish = values[end_index]
                    distinct_val = np.unique(values)
                    meta = meta = [start,finish,'True',average_freq,len(distinct_val)] 
                    bins.append(meta)
                    start_index = end_index + 1
                    end_index = end_index + average_freq
                    values = []
                    if(end_index > len(data_sorted_numpy)):
                        end_index = len(data_sorted_numpy)
            partitions[data_sorted.columns[i]] = bins
        num_bins = original_bins
        hist_bins = original_hist_bins
        start_index = 0
        end_index = average_freq

    state = {'partitions':partitions, 'total': total, 'encoder': encoder_map, 'categorical_variables': categorical_variables, 'hash_functions': hash_functions, 'sketches': attribute_sketches, 'num_bins': num_bins}
    dur_ms = (time.time() - start_time) 
    L.info(f"Time taken to build: {dur_ms} s")
    return state

def test_sketches_hist(seed: int, dataset: str, version: str, workload: str, params: Dict[str, Any], overwrite: bool) -> None:
    """
    params:
        version: the version of table that the histogram is built from, might not be the same with the one we test on
        num_bins: maximum number of partitions
    """
    # prioriy: params['version'] (draw sample from another dataset) > version (draw and test on the same dataset)
    table = load_table(dataset, params.get('version') or version)

    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}-sketches_Hist{params['num_bins']}.pkl"

    if model_file.is_file():
        L.info(f"{model_file} already exists, directly load and use")
        with open(model_file, 'rb') as f:
            state = pickle.load(f)
    else:
        L.info(f"Construct sketches_Hist with at most {params['num_bins']} bins...")
        state = construct_bins(table, params['num_bins'],100)
        with open(model_file, 'wb') as f:
            pickle.dump(state, f, protocol=PKL_PROTO)
        L.info(f"sketches_Hist saved to {model_file}")

    # partitions = attribute_bins
    estimator = Sketches_Hist(state, params['num_bins'], table)
    L.info(f"Built sketches_Hist estimator: {estimator}")

    run_test(dataset, version, workload, estimator, overwrite)
