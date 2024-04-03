#! /usr/bin/env python3

import argparse
import ast
import csv
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import json
import os

COIN = 100000000

def satoshi_round(amount):
    return Decimal(amount).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

def to_sat(amount):
    return int(amount * COIN)

def to_coin(amount):
    return satoshi_round(amount / COIN)

parser = argparse.ArgumentParser("Calculate the basic fee statistics for a scenario file")
parser.add_argument("scenario", help="CSV file to import scenario from")
parser.add_argument("--min_bin_size", dest="min_bin_size", type=int, default=1000, help="minimum bin size (default=1000)")
parser.add_argument("--max_bin_chance", dest="max_bin_chance", type=float, default=0.05, help="maximum probability per bin (default=0.05)")
args = parser.parse_args()

with open(args.scenario) as scenario_file:
    new_f = open(f"feerates-{os.path.splitext(os.path.basename(args.scenario))[0]}.json", "w")
    scenarioreader = csv.reader(scenario_file, delimiter=',', quotechar='"')
    next(scenarioreader)
    tx_size_vbytes = 111 # p2tr, 1 input, 1 output: https://bitcoinops.org/en/tools/calc-size/
    tx_size_vbytes += 4 # why do I need to add this to match bitcoind.

    encoder = json.JSONEncoder(indent="  ")
    count = 0
    min_total_fees = 0
    feerates = {}

    for row in scenarioreader:
        spend = ast.literal_eval(row[0])
        feerate = ast.literal_eval(row[1])
        tx_fee = satoshi_round(feerate * (tx_size_vbytes/args.min_bin_size)) # 1 input, 1 output
        if spend < 0:
            min_total_fees += tx_fee
        bin = round(to_sat(feerate)/args.min_bin_size)
        if bin not in feerates:
            feerates[bin] = 1
        else:
            feerates[bin] = feerates[bin]+1
        count += 1
        
    feerates = dict(sorted(feerates.items()))
    feereates_list = []
    current_feerate = {}
    current_feerate = {'start': args.min_bin_size, 'end': 0, 'probability': 0}
    current_probability = 0
    for bin in feerates:
         current_probability += feerates[bin]/count
         if (current_probability >= args.max_bin_chance):
            current_feerate['end'] = bin*args.min_bin_size-1
            current_feerate['probability'] = round(current_probability,2)
            feereates_list.append(current_feerate)
            current_probability = current_probability - current_feerate['probability']
            current_feerate = {'start': bin*args.min_bin_size, 'end': 0, 'probability': current_probability}
    current_feerate['end'] = bin*args.min_bin_size-1
    current_feerate['probability'] = round(current_probability, 2)
    feereates_list.append(current_feerate)
    new_f.write(encoder.encode(feereates_list))

    # compute min max bucket for each target    
    min_fee = to_sat(satoshi_round(to_coin(feereates_list[0]['start']) * Decimal(tx_size_vbytes/args.min_bin_size)))
    max_fee = to_sat(satoshi_round(to_coin(feereates_list[-1]['end']) * Decimal(tx_size_vbytes/args.min_bin_size)))
    print(f"min_max_fee= ({min_fee}, {max_fee})")
    print(f"min_total_fees= {min_total_fees}")
