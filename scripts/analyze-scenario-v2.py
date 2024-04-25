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
parser.add_argument("--bucket_refill_feerate", dest="bucket_refill_feerate", type=int, default=30000, help="bucket refill rate, sats per kvbyte (default=30000)")
args = parser.parse_args()

with open(args.scenario) as scenario_file:
    new_f = open(f"feerates-{os.path.splitext(os.path.basename(args.scenario))[0]}.json", "w")
    scenarioreader = csv.reader(scenario_file, delimiter=',', quotechar='"')
    next(scenarioreader)
    tx_size_changeless_vbytes = 111 # p2tr, 1 input, 1 output: https://bitcoinops.org/en/tools/calc-size/
    tx_size_changeless_vbytes += 4 # why do I need to add this to match bitcoind.
    tx_change_size = 43

    encoder = json.JSONEncoder(indent="  ")
    count = 0
    min_total_fees = 0

    feerate_spends = []
    for row in scenarioreader:
        spend = ast.literal_eval(row[0])
        if spend < 0:
            feerate = ast.literal_eval(row[1])
            tx_fee = satoshi_round(feerate * (tx_size_changeless_vbytes/1000.0)) # 1 input, 1 output
            min_total_fees += tx_fee
            feerate_spends.append({"feerate":to_sat(feerate), "spend":to_sat(spend*-1)})

    reverse_sorted_feerate_spends = sorted(feerate_spends, key=lambda d: d['feerate'], reverse=True)

    feerate_bins = [{"start": 0, "end": reverse_sorted_feerate_spends[0]["feerate"]}]
    count = 0
    total_count = 0
    for feerate_spend in reverse_sorted_feerate_spends:
        feerate = feerate_spend["feerate"]
        target_changeless_fee = feerate_bins[-1]["end"]*tx_size_changeless_vbytes/1000.0
        current_change_fee = feerate*tx_change_size/1000.0
        current_changeless_tx_fee = feerate*tx_size_changeless_vbytes/1000.0
        if target_changeless_fee - current_changeless_tx_fee > current_change_fee:
            feerate_bins[-1]["start"] = feerate + 1
            feerate_bins[-1]["probability"] = Decimal(count / len(reverse_sorted_feerate_spends)).quantize(Decimal('0.001'))
            feerate_bins.append({"start": 0, "end": feerate})
            count = 0
        count += 1
        total_count += 1
    feerate_bins[-1]["start"] = 0
    feerate_bins[-1]["probability"] = Decimal(count / len(reverse_sorted_feerate_spends)).quantize(Decimal('0.001'))
    feerate_bins = sorted(feerate_bins, key=lambda d: d['start'])
    assert(Decimal(sum(item['probability'] for item in feerate_bins)).quantize(Decimal('0.01')) == 1.0)

    # for example, use this refill feerate
    bucket_refill_feerate = args.bucket_refill_feerate
    for bin in feerate_bins:
        bin["count"] = 0
        bin["max_count"] = 0

    # compute how many total buckets we need to not empty any bucket
    for feerate_spend in feerate_spends:
        if feerate_spend["feerate"] < bucket_refill_feerate:
            for bin in feerate_bins:
                bin["count"] = 0

        for bin in feerate_bins:
            target_changeless_fee = bin["end"]*tx_size_changeless_vbytes/1000.0
            current_change_fee = feerate*tx_change_size/1000.0
            current_changeless_tx_fee = feerate*tx_size_changeless_vbytes/1000.0
            if target_changeless_fee - current_changeless_tx_fee > current_change_fee:
                bin["count"] += 1
                if bin["count"] > bin["max_count"]:
                    bin["max_count"] = bin["count"]

    # compute max_count / feerate sat/kvbyte range
    for bin in feerate_bins:
        if bin["max_count"] > 0:
            bin["feerate_per_count"]=(bin["end"]-bin["start"])/bin["max_count"]
        else:
            bin["feerate_per_count"]=0

    feerate_bins2 = []
    for bin in feerate_bins:
        target_count = int(round(bin["max_count"]/1000.0 + 0.5))*10
        if target_count > 0:
            feerate_bins2.append({"feerate": bin["end"], "target_count": target_count})
    
    targets = {}
    targets['bucket_refill_feerate'] = bucket_refill_feerate
    targets['feerates_satoshis_per_kvbyte'] = feerate_bins2
    new_f.write(encoder.encode(targets))

    # compute min max bucket for each target    
    min_fee = to_sat(satoshi_round(to_coin(targets['feerates_satoshis_per_kvbyte'][0]['feerate']) * Decimal(tx_size_changeless_vbytes/1000)))
    max_fee = to_sat(satoshi_round(to_coin(targets['feerates_satoshis_per_kvbyte'][-1]['feerate']) * Decimal(tx_size_changeless_vbytes/1000)))
    print(f"min_max_feerate_per_kvbyte = ({targets['feerates_satoshis_per_kvbyte'][0]['feerate']}, {targets['feerates_satoshis_per_kvbyte'][-1]['feerate']})")
    print(f"min_max_fee = ({min_fee}, {max_fee})")
    print(f"min_total_fees = {min_total_fees}")
    print(f"bucket_refill_feerate= {targets['bucket_refill_feerate']}")
