#! /usr/bin/env python3

import argparse
import ast
import csv
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import itertools
import json
import os
from random import random
from statistics import mean, stdev

COIN = 100000000

def satoshi_round(amount):
    return Decimal(amount).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

def to_sat(amount):
    return int(amount * COIN)

def to_coin(amount):
    return satoshi_round(amount / COIN)

def get_next_feerate(feerates_satoshis_per_kvbyte):
    prob = random()
    accum = 0
    for feerate_bin in feerates_satoshis_per_kvbyte:
        range = feerate_bin['end'] - feerate_bin['start']
        if accum > prob:
            return feerate_bin['start'] + range * random()
        else:
            accum += feerate_bin['probability']

    return feerate_bin['start'] + range * random()

def target_from_amount(amount, utxo_targets, min_fee, max_fee): 
    for bucket in range(len(utxo_targets)):
        start_amount = utxo_targets[bucket]['start_satoshis'] + min_fee
        end_amount = utxo_targets[bucket]['start_satoshis'] + max_fee
        if amount >= start_amount and amount <= end_amount:
            return bucket
    return -1

def next_change_target(utxo_targets, target_counts, feerates_satoshis_per_kvbyte, tx_size_vbytes):
    min_bucket = -1
    min_capacity = None
    min_fee = to_sat(satoshi_round(to_coin(feerate_list[0]['start']) * Decimal(tx_size_vbytes/1000)))
    max_fee = to_sat(satoshi_round(to_coin(feerate_list[-1]['end']) * Decimal(tx_size_vbytes/1000)))
    for i in range(len(utxo_targets)):
        capacity = float(target_counts[i]) / utxo_targets[i]['target_utxo_count']
        if min_capacity is None or capacity < min_capacity:
            min_bucket = i
            min_capacity = capacity
    next_fee = get_next_feerate(feerates_satoshis_per_kvbyte)
    amount = utxo_targets[min_bucket]['start_satoshis'] + to_sat(satoshi_round(to_coin(next_fee) * Decimal(tx_size_vbytes/1000)))
    assert(min_bucket == target_from_amount(amount, utxo_targets, min_fee, max_fee))
    return min_bucket, min_capacity, amount

def split_residual_change(change_outputs, utxo_targets, target_counts, change_fee_sat, min_fee, max_fee, bucket_max):
    bucket = target_from_amount(change_outputs[-1], utxo_targets, min_fee, max_fee)
    if change_outputs[-1] <= bucket_max and bucket == -1:
        # split residual change that is less than the largest possible bucket value into useful buckets
        numbers = []
        tmp_target_counts = target_counts
        for _ in range(20):
            bucket, _, target_amount = next_change_target(utxo_targets, tmp_target_counts, feerates_satoshis_per_kvbyte, tx_size_vbytes)
            tmp_target_counts[bucket] += 1
            numbers.append(target_amount + change_fee_sat)
        target = change_outputs[-1]
        result = [seq for i in range(len(numbers), 0, -1)
        for seq in itertools.combinations(numbers, i)
        if sum(seq) > target*0.98 and sum(seq) <= target]
        best_combination = [change_outputs[-1]]
        best_combination_diff = change_outputs.pop()
        for seq in result:
            if (target - sum(seq) < best_combination_diff):
                best_combination = seq
                best_combination_diff = target - sum(seq)
        for output_amount in best_combination:
            change_outputs.append(output_amount - change_fee_sat)

    return change_outputs

# split change output to replentish utxo_targets
def split_to_change_targets(change_amount, change_target, utxo_targets, target_counts, feerates_satoshis_per_kvbyte, change_fee_sat, max_outputs):
    assert(change_amount > change_fee_sat + change_target)
    # compute initial change target
    change_outputs = [change_target]
    change_amount -= change_fee_sat + change_target
    bucket = target_from_amount(change_target, utxo_targets, min_fee, max_fee)
    assert(bucket != -1)
    target_counts[bucket] += 1

    while (max_outputs < 0 or len(change_outputs) < max_outputs):
        # compute next target amount
        bucket, _, target_amount = next_change_target(utxo_targets, target_counts, feerates_satoshis_per_kvbyte, tx_size_vbytes)
        if (target_counts[bucket] >= utxo_targets[bucket]['target_utxo_count']):
            break
        if (change_amount < change_fee_sat + target_amount):
            break
        change_amount -= change_fee_sat + target_amount
        target_counts[bucket] += 1
        # add initial change output
        change_outputs.append(target_amount)

    # remaining change goes into the last change output 
    change_outputs[-1] += change_amount

    return change_outputs

def compute_target_counts(utxos, pending_txs, utxo_targets, feerates_satoshis_per_kvbyte, tx_size_vbytes):
    min_fee = to_sat(satoshi_round(to_coin(feerates_satoshis_per_kvbyte[0]['start']) * Decimal(tx_size_vbytes/1000)))
    max_fee = to_sat(satoshi_round(to_coin(feerates_satoshis_per_kvbyte[-1]['end']) * Decimal(tx_size_vbytes/1000)))
    target_counts = [0]*len(utxo_targets)
    for pending_tx in pending_txs:
        for utxo in pending_tx:
            bucket = target_from_amount(utxo, utxo_targets, min_fee, max_fee)
            if bucket > -1:
                target_counts[bucket] += 1

    for utxo in utxos:
        bucket = target_from_amount(to_sat(utxo), utxo_targets, min_fee, max_fee)
        if bucket > -1:
            target_counts[bucket] += 1
    return target_counts

def spend_from_large_non_bucket(amount_needed, utxos, utxo_targets, min_fee, max_fee):
    reverse_sorted_utxos = sorted(utxos, reverse=True)
    for amount in reverse_sorted_utxos:
        change = amount - amount_needed
        if to_sat(change) > bucket_max:
            break
        if amount < bucket_max:
            # no large unspent utxo found
            break
    
    spent_bucket = target_from_amount(to_sat(amount), utxo_targets, min_fee, max_fee)
    assert(spent_bucket == -1)
    return amount, spent_bucket

def check_fees(feerate, tx_size_vbytes, total_fees, outputs, change_fee):
    change_fee = (Decimal(feerate) * Decimal(43) / Decimal(1000.0)).quantize(Decimal('0.00000001'), ROUND_UP)
    tx_fee = (Decimal(feerate) * Decimal(tx_size_vbytes) / Decimal(1000.0)).quantize(Decimal('0.00000001'), ROUND_UP)
    tx_fee_total = tx_fee + len(outputs)*change_fee
    tx_size = tx_size_vbytes + len(outputs)*43
    feerate1 = satoshi_round(total_fees/Decimal(tx_size/1000))
    assert(feerate1 >= satoshi_round(feerate))
    assert(feerate1 < satoshi_round(feerate*1.05))

parser = argparse.ArgumentParser("Calculate the basic statistics for a scenario file")
parser.add_argument("scenario", help="CSV file to import scenario from")
parser.add_argument("targets", help="JSON file to import target data from")
parser.add_argument("repeat", help="number of times to repeat the test scenario")
parser.add_argument("--excess_percent", dest="excess_percent", type=float, default=0.05, help="percent amount target can be increased over requested target spend (default: 0.05)")
args = parser.parse_args()

decoder = json.JSONDecoder()

result_filename = f"results-{os.path.splitext(os.path.basename(args.targets))[0]}-excess{args.excess_percent}.csv"
results_f = open(result_filename, "w")
results = csv.DictWriter(
    results_f,
    [
        "target_file",
        "bucket_refill_feerate",
        "minimum_total_fees",
        "simulated_total_fees",
        "simulated_total_excess",
        "utxos_count",
        "target_counts",
        "total_sent_value",
        "total_utxo_value",
        "total_pending_utxo_value",
        "total_utxo_value+total_sent_value+total_fees+total_pending_utxo_value",
        "total_funding_value",
        "received_utxos",
        "received_utxos_value",
        "total_changeless",
        "total_with_change",
        "utxos_not_in_buckets",
        "utxos_in_buckets",
        "small_change_outputs_not_in_buckets",
        "small_utxos_not_in_buckets",
        "failure_count",
        "failures"
    ],
    )
results.writeheader()

tx_size_vbytes = 111 # p2tr, 1 input, 1 output: https://bitcoinops.org/en/tools/calc-size/
tx_size_vbytes += 4 # why do I need to add this to match bitcoind.

ftargets = open(args.targets)
decoder = json.JSONDecoder()
encoder = json.JSONEncoder()
text = ftargets.read()
json_dec = decoder.decode(text)
utxo_targets = json_dec['buckets']
feerates_satoshis_per_kvbyte = json_dec['feerates_satoshis_per_kvbyte']
total_feerate_probability = 0
for feerate_probability in feerates_satoshis_per_kvbyte:
    total_feerate_probability += feerate_probability['probability']
assert(round(total_feerate_probability,6) == 1.0)
# compute min max bucket for each target    
feerate_list = list(feerates_satoshis_per_kvbyte)
min_fee = to_sat(satoshi_round(to_coin(feerate_list[0]['start']) * Decimal(tx_size_vbytes/1000)))
max_fee = to_sat(satoshi_round(to_coin(feerate_list[-1]['end']) * Decimal(tx_size_vbytes/1000)))
for i in range(len(utxo_targets)):
    utxo_targets[i]['start_total'] = min_fee + utxo_targets[i]['start_satoshis']
    utxo_targets[i]['end_total'] = max_fee + utxo_targets[i]['start_satoshis']
bucket_max = utxo_targets[-1]['start_satoshis']+max_fee
results_stats = []

for _ in range(int(args.repeat)):
    with open(args.scenario) as scenario_file:
        results_stats.append({"target_file": args.targets})
        results_stats[-1]['bucket_refill_feerate'] = json_dec['bucket_refill_feerate']
        count = 0
        min_total_fees = 0
        sim_total_fees = 0
        feerates = {}
        utxos = [satoshi_round(0.05)]*255
        total_funding_value = sum(utxos)
        total_sent_value = 0
        with_change_txs = []
        changeless_txs = []
        pending_txs = []
        received_utxos = []
        small_change_outputs_not_in_buckets = []
        # cost to add a taproot output
        failures = []
        scenarioreader = csv.reader(scenario_file, delimiter=',', quotechar='"')
        next(scenarioreader)
        for row in scenarioreader:
            spend = ast.literal_eval(row[0])
            feerate = ast.literal_eval(row[1])
            # 1 input, 1 output
            tx_fee = Decimal(feerate * (tx_size_vbytes/1000)).quantize(Decimal('0.00000001'), rounding=ROUND_UP)
            change_fee = Decimal(feerate * (43/1000.0)).quantize(Decimal('0.00000001'), rounding=ROUND_UP)
            if spend < 0:
                min_total_fees += tx_fee
            bin = round(to_sat(feerate)/1000)
            if bin not in feerates:
                feerates[bin] = 1
            else:
                feerates[bin] = feerates[bin]+1
            count += 1

            # recompute target counts
            target_counts = compute_target_counts(utxos, pending_txs, utxo_targets, feerates_satoshis_per_kvbyte, tx_size_vbytes)

            # track hypothetical UTXOs
            if spend > 0:
                utxos.append(satoshi_round(spend))
                total_funding_value += satoshi_round(spend)
                received_utxos.append(spend)
            else:
                success = False
                spend = satoshi_round(spend * -1)
                sorted_utxos = sorted(utxos)
                amount_needed = spend + tx_fee
                target_bucket = target_from_amount(to_sat(amount_needed), utxo_targets, min_fee, max_fee)
                for amount in sorted_utxos:
                    spent_bucket = target_from_amount(to_sat(amount), utxo_targets, min_fee, max_fee)
                    if satoshi_round(amount) > amount_needed:
                        total_fees = satoshi_round(amount) - amount_needed
                        if total_fees > spend*Decimal(args.excess_percent):
                            # select largest unused utxo instead
                            #amount, spent_bucket = spend_from_large_non_bucket(amount_needed, sorted_utxos, utxo_targets, min_fee, max_fee)
                            amount = sorted_utxos[-1]
                            if amount >= amount_needed:
                                # found solution with change
                                change_amount = amount - amount_needed
                                min_bucket, min_capacity, change_target = next_change_target(utxo_targets, target_counts, feerates_satoshis_per_kvbyte, tx_size_vbytes)
                                # change_target = to_sat(amount_needed)
                                if feerate < to_coin(results_stats[-1]['bucket_refill_feerate']) and to_sat(change_amount) > to_sat(change_fee) + change_target:
                                    # split to replace spent target and refill as many buckets as possible
                                    outputs_sats = split_to_change_targets(to_sat(change_amount), change_target, utxo_targets, target_counts, feerates_satoshis_per_kvbyte, to_sat(change_fee), -1)
                                else:
                                    outputs_sats = [to_sat(change_amount - change_fee)]
                                #outputs_sats = split_residual_change(outputs_sats, utxo_targets, target_counts, to_sat(change_fee), min_fee, max_fee, bucket_max)

                                outputs = []
                                for output_sats in outputs_sats:
                                    outputs.append(to_coin(output_sats))
                                    out_bucket = target_from_amount(output_sats, utxo_targets, min_fee, max_fee)
                                    if out_bucket == -1 and output_sats < bucket_max:
                                        small_change_outputs_not_in_buckets.append(output_sats)
                                    
                                total_fees = amount - (spend + sum(outputs))
                                check_fees(feerate, tx_size_vbytes, total_fees, outputs, change_fee)
                                pending_txs.append(outputs)
                                total_sent_value += Decimal(spend)
                                with_change_txs.append({"target_bucket": target_bucket, "spent_bucket": spent_bucket, "utxo_amount": Decimal(amount), "feerate": feerate, "excess": satoshi_round(Decimal(amount) - amount_needed)})
                                success = True
                                break
                            else:
                                failures.append({"spent_bucket": spent_bucket})
                                success = False
                                break
                        else:
                            # found changeless solution
                            pending_txs.append([])
                            total_fees = tx_fee
                            total_sent_value += Decimal(amount - tx_fee)
                            assert(spent_bucket == target_bucket or spent_bucket == -1)
                            changeless_txs.append({"target_bucket": target_bucket, "utxo_amount": Decimal(amount), "feerate": feerate, "excess": satoshi_round(Decimal(amount) - amount_needed)})
                            success = True
                            break
                if success:
                    sim_total_fees += total_fees
                    utxos.remove(amount)
                else:
                    failures[-1]["spend"] = to_coin(spend)
                    failures[-1]["tx_fee"] = tx_fee
                    failures[-1]["target_bucket"] = target_bucket

                # add pending change utxos
                if len(pending_txs) >= 50:
                    confirmed_utxos = pending_txs.pop(0)
                    for utxo in confirmed_utxos:
                        utxos.append(utxo)

        total_pending_utxo_value = 0
        for tx in pending_txs:
            for utxo in tx:
                total_pending_utxo_value += utxo

        sim_total_excess = 0
        for tx in changeless_txs:
            sim_total_excess += tx["excess"]

        utxos_not_in_buckets = []
        utxos_in_buckets = []
        for utxo in utxos:
            bucket = target_from_amount(to_sat(utxo), utxo_targets, min_fee, max_fee)
            if bucket > -1:
                utxos_in_buckets.append({"bucket": bucket, "amount": to_sat(utxo)})
            else:
                utxos_not_in_buckets.append(to_sat(utxo))
                

        for tx in pending_txs:
            for utxo in tx:
                bucket = target_from_amount(utxo, utxo_targets, min_fee, max_fee)
                if bucket > -1:
                    utxos_in_buckets.append({"bucket": bucket, "amount": utxo})
                else:
                    utxos_not_in_buckets.append(utxo)

        small_utxos_not_in_bucket = []
        for utxo in utxos_not_in_buckets:
            if utxo < bucket_max:
                small_utxos_not_in_bucket.append(utxo)

        target_counts = compute_target_counts(utxos, pending_txs, utxo_targets, feerates_satoshis_per_kvbyte, tx_size_vbytes)
        
        results_stats[-1]["minimum_total_fees"]=min_total_fees
        results_stats[-1]["simulated_total_fees"]=sim_total_fees
        results_stats[-1]["simulated_total_excess"]=sim_total_excess
        results_stats[-1]["utxos_count"]=len(utxos)
        results_stats[-1]["target_counts"]=target_counts
        results_stats[-1]["total_sent_value"]=total_sent_value
        results_stats[-1]["total_utxo_value"]=sum(utxos)
        results_stats[-1]["total_pending_utxo_value"]=total_pending_utxo_value
        results_stats[-1]["total_utxo_value+total_sent_value+total_fees+total_pending_utxo_value"]=sum(utxos)+total_sent_value+sim_total_fees+total_pending_utxo_value
        results_stats[-1]["total_funding_value"]=total_funding_value
        results_stats[-1]["received_utxos"]=len(received_utxos)
        results_stats[-1]["received_utxos_value"]=sum(received_utxos)
        results_stats[-1]["total_changeless"]=len(changeless_txs)
        results_stats[-1]["total_with_change"]=len(with_change_txs)
        results_stats[-1]["utxos_not_in_buckets"]=len(utxos_not_in_buckets)
        results_stats[-1]["utxos_in_buckets"]=len(utxos_in_buckets)
        results_stats[-1]["small_change_outputs_not_in_buckets"]=len(small_change_outputs_not_in_buckets)
        results_stats[-1]["small_utxos_not_in_buckets"]=len(small_utxos_not_in_bucket)
        results_stats[-1]["failure_count"]=len(failures)
        results_stats[-1]["failures"]=str(failures)
        results.writerow(results_stats[-1])
        print(f"{len(results_stats)}. simulated_total_fees: {results_stats[-1]['simulated_total_fees']}")

sorted_results_stats = {"target_file": "Sorted"}
mean_results_stats = {"target_file": "Mean"}
stdev_results_stats = {"target_file": "Stdev"}
for k in results_stats[0].keys():
    if not isinstance(results_stats[0][k], str) and not isinstance(results_stats[0][k], list):
        sorted_results_stats[k] = []
        for result in results_stats:
            sorted_results_stats[k].append(result[k])
        sorted_results_stats[k] = sorted(sorted_results_stats[k])
        mean_results_stats[k] = satoshi_round(mean(sorted_results_stats[k]))
        stdev_results_stats[k] = satoshi_round(stdev(sorted_results_stats[k]))

results.writerow(mean_results_stats)
results.writerow(stdev_results_stats)