#! /usr/bin/env python3

import argparse
import ast
import csv
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import numpy as np
import json
import os

COIN = 100000000

def satoshi_round(amount):
    return Decimal(amount).quantize(Decimal('0.00000001'))

def to_sat(amount):
    return int(amount * COIN)

def to_coin(amount):
    return Decimal(amount / COIN).quantize(Decimal('0.00000001'))

def next_change_target(target_counts, max_amount):
    min_capacity = None
    sorted_targets = sorted(target_counts, key=lambda d: d['target_amount'])
    for target in sorted_targets:
        capacity = float(target["count"]) / (target["target_count"])
        if min_capacity is None or capacity < min_capacity and target["target_amount"] <= max_amount:
            min_amount = target["target_amount"]
            min_capacity = capacity
    return min_capacity, min_amount

# split change output to replentish utxo_targets
def split_to_change_targets(change_amount, change_target, target_counts, change_fee_sat, max_outputs):
    if change_amount < change_fee_sat + change_target:
        # insufficient change for change_target
        return [change_amount - change_fee_sat]
    
    # compute initial change target
    change_outputs = [change_target]
    change_amount -= change_fee_sat + change_target
    assert(is_target(change_target, target_counts))
    increment_target(change_target, target_counts)

    while (max_outputs < 0 or len(change_outputs) < max_outputs):
        # compute next target amount
        capacity, target_amount = next_change_target(target_counts, change_amount - change_fee_sat)
        if (capacity >= 1.0):
            break
        if (change_amount < change_fee_sat + target_amount):
            break
        change_amount -= change_fee_sat + target_amount
        increment_target(target_amount, target_counts)
        # add initial change output
        change_outputs.append(target_amount)

    # sanity check
    for change_output in change_outputs:
        assert(is_target(change_output, target_counts))
        assert(target_capacity(to_sat(change_output), target_counts) <= 1.0)

    # remaining change goes into the last change output 
    change_outputs[-1] += change_amount

    return change_outputs

def get_nearest_target(amount_sats, sorted_target_counts):
    # target counts must be sorted, not reverse sorted
    for target in sorted_target_counts:
        excess = target["target_amount"] - amount_sats 
        # excess must be within tx_fee of nearest target amount
        if amount_sats <= target["target_amount"] and excess < target["tx_fee"]:
            return target
    return None

def compute_feerate_target_buckets(utxo_targets, feerates_satoshis_per_kvbyte):
    buckets = []
    for feerate in feerates_satoshis_per_kvbyte:
        for target in utxo_targets:
            tx_fee = to_sat(satoshi_round(to_coin(feerate["feerate"]) * Decimal(tx_size_vbytes/1000)))
            amount_needed = target['start_satoshis'] + tx_fee
            buckets.append({"target_amount": amount_needed, "count": 0, "target_count": int(feerate["target_count"]*target["target_utxo_percent"]), "tx_fee": tx_fee})
    sorted_buckets = sorted(buckets, key=lambda d: d['target_amount'])
    return sorted_buckets

def compute_target_counts(utxos, pending_txs, utxo_targets, feerates_satoshis_per_kvbyte):
    sorted_target_counts = compute_feerate_target_buckets(utxo_targets, feerates_satoshis_per_kvbyte)

    # count confirmed utxos into nearest (or greater) buckets
    for utxo_amount in utxos:
        target = get_nearest_target(to_sat(utxo_amount), sorted_target_counts)
        if target: 
            if target["target_amount"] == to_sat(utxo_amount):
                target["count"] += 1
            #else:
            #    print(f"utxo near miss, target={target['target_amount']}, utxo_amount={to_sat(utxo_amount)}")

    # count pending utxos into nearest (or greater) buckets
    for pending_tx in pending_txs:
        for utxo_amount in pending_tx:
            target = get_nearest_target(to_sat(utxo_amount), sorted_target_counts)
            if target:
                if target["target_amount"] == to_sat(utxo_amount):
                    target["count"] += 1
                #else:
                #    print(f"pending utxo near miss, target={target['target_amount']}, utxo_amount={to_sat(utxo_amount)}")

    reverse_sorted_target_counts = sorted(sorted_target_counts, key=lambda d: d['target_amount'], reverse=True)

    return reverse_sorted_target_counts

def is_target(amount, target_counts):
    found = True
    for target in target_counts:
        if amount == target["target_amount"]:
            return True
    return False

def target_capacity(amount, target_counts):
    for target in target_counts:
        if target["target_amount"] == amount:
            return float(target["count"]) / (target["target_count"])
    return 0.0

def increment_target(amount, target_counts1):
    for target in target_counts1:
        if amount == target["target_amount"]:
            target["count"] += 1
            return
    assert(0)

def spend_from_large_non_bucket(amount_needed, sorted_utxos, reverse_sorted_target_counts):
    sorted_non_buckets = np.fromiter((x for x in sorted_utxos if not is_target(to_sat(x), reverse_sorted_target_counts)), dtype=sorted_utxos.dtype)
    idx = sorted_non_buckets.searchsorted(amount_needed, side='right')
    if idx < len(sorted_non_buckets):
        return sorted_non_buckets[idx] 
    return -1

def check_fees(feerate, tx_size_vbytes, total_fees, outputs, change_fee):
    change_fee = (Decimal(feerate) * Decimal(change_size_vbytes) / Decimal(1000.0)).quantize(Decimal('0.00000001'), ROUND_UP)
    tx_fee = (Decimal(feerate) * Decimal(tx_size_vbytes) / Decimal(1000.0)).quantize(Decimal('0.00000001'), ROUND_UP)
    tx_size = tx_size_vbytes + len(outputs)*change_size_vbytes
    feerate1 = satoshi_round(total_fees/Decimal(tx_size/1000))
    assert(abs(tx_fee + len(outputs)*change_fee - total_fees) <= 1)
    assert(feerate1*Decimal(1.01) >= satoshi_round(feerate))
    assert(feerate1 < satoshi_round(feerate*1.05))

parser = argparse.ArgumentParser("Calculate the basic statistics for a scenario file")
parser.add_argument("scenario", help="CSV file to import scenario from")
parser.add_argument("targets", help="JSON file to import target data from")
args = parser.parse_args()

decoder = json.JSONDecoder()

result_filename = f"results-{os.path.splitext(os.path.basename(args.scenario))[0]}-{os.path.splitext(os.path.basename(args.targets))[0]}.csv"
results_f = open(result_filename, "w")
results = csv.DictWriter(
    results_f,
    [
        "target_file",
        "bucket_refill_feerate",
        "extra_fees_percent",
        "excess_percent",
        "changeless_utxos_percent",
        "minimum_total_fees",
        "simple_total_fees",
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
        "changeless_misses",
        "failure_count",
        "failures"
    ],
    )
results.writeheader()

tx_size_vbytes = 111 # p2tr, 1 input, 1 output: https://bitcoinops.org/en/tools/calc-size/
tx_size_vbytes += 4 # why do I need to add this to match bitcoind.
change_size_vbytes = 43 # for each additional change output

ftargets = open(args.targets)
decoder = json.JSONDecoder()
encoder = json.JSONEncoder()
text = ftargets.read()
json_dec = decoder.decode(text)
utxo_targets = json_dec['buckets']
feerates_satoshis_per_kvbyte = json_dec['feerates_satoshis_per_kvbyte']
bucket_refill_feerate = json_dec['bucket_refill_feerate']

# compute min max bucket for each target    
feerate_list = list(feerates_satoshis_per_kvbyte)
min_fee = to_sat(satoshi_round(to_coin(feerate_list[0]['feerate']) * Decimal(tx_size_vbytes/1000)))
max_fee = to_sat(satoshi_round(to_coin(feerate_list[-1]['feerate']) * Decimal(tx_size_vbytes/1000)))
bucket_max = utxo_targets[-1]['start_satoshis']+max_fee

with open(args.scenario) as scenario_file:
    results_stats = {"target_file": args.targets}
    feerates = {}
    utxos = [satoshi_round(0.05)]*255
    total_funding_value = sum(utxos)
    total_sent_value = 0
    with_change_txs = []
    changeless_txs = []
    last_changeless_txs = 0
    last_total_txs = 0
    pending_txs = []
    received_utxos = []
    small_change_outputs_not_in_buckets = []
    # cost to add a taproot output
    failures = []
    scenarioreader = csv.reader(scenario_file, delimiter=',', quotechar='"')

    min_total_fees = 0
    simple_total_fees = 0
    sim_total_fees = 0
    feerate_spends = []
    for row in scenarioreader:
        spend = ast.literal_eval(row[0])
        feerate = ast.literal_eval(row[1])
        feerate_spends.append({"feerate": feerate, "spend": spend})
        # 1 input, 1 output
        tx_fee = Decimal(feerate * (tx_size_vbytes/1000)).quantize(Decimal('0.00000001'))
        change_fee = Decimal(feerate * (change_size_vbytes/1000.0)).quantize(Decimal('0.00000001'), rounding=ROUND_UP)
        if spend < 0:
            min_total_fees += tx_fee
            simple_total_fees += tx_fee + change_fee # 1 in 2 out (funding + change)
        bin = round(to_sat(feerate)/1000)
        if bin not in feerates:
            feerates[bin] = 1
        else:
            feerates[bin] = feerates[bin]+1

    print(f"minimum_total_fees: {min_total_fees} 1.0x")
    print(f"simple_total_fees: {simple_total_fees} {Decimal(simple_total_fees/min_total_fees).quantize(Decimal('0.01'))}x")
    
    count = 0
    min_total_fees = 0
    simple_total_fees = 0
    sim_total_fees = 0
    changeless_misses = compute_feerate_target_buckets(utxo_targets, feerates_satoshis_per_kvbyte)
    for target in changeless_misses:
        target["changeless_misses"] = 0

    for feerate_spend in feerate_spends:
        feerate = feerate_spend["feerate"]
        spend = feerate_spend["spend"]
        tx_fee = Decimal(feerate * (tx_size_vbytes/1000)).quantize(Decimal('0.00000001'))
        change_fee = Decimal(feerate * (change_size_vbytes/1000.0)).quantize(Decimal('0.00000001'), rounding=ROUND_UP)

        # recompute target counts
        target_counts = compute_target_counts(utxos, pending_txs, utxo_targets, feerates_satoshis_per_kvbyte)

        # track hypothetical UTXOs
        if spend > 0:
            utxos.append(satoshi_round(spend))
            total_funding_value += satoshi_round(spend)
            received_utxos.append(spend)
        else:
            min_total_fees += tx_fee
            simple_total_fees += tx_fee + change_fee # 1 in 2 out (funding + change)
            changeless = False
            success = False
            spend = satoshi_round(spend * -1)
            sorted_utxos = np.sort(np.unique(np.array(utxos)))
            amount_needed = spend + tx_fee
            capacity,_ = next_change_target(target_counts, sorted_utxos[-1])
            refill = feerate < to_coin(bucket_refill_feerate) and capacity < 0.7

            # find smallest utxo that is greater than the amount needed to fund the spend
            if refill:
                utxo_amount = utxo_amount = sorted_utxos[-1]
            else:
                idx = sorted_utxos.searchsorted(amount_needed, side='right')
                if idx < len(sorted_utxos):
                    utxo_amount = sorted_utxos[idx]
                    excess = utxo_amount - amount_needed
                    
                    # check if nearest utxo can be used without change
                    if excess > change_fee and to_sat(tx_fee) > min_fee:
                        # not found in our current set of confirmed utxos
                        target = get_nearest_target(to_sat(amount_needed), changeless_misses)
                        if target:
                            assert(not target["target_amount"] in sorted_utxos)
                            assert(target["target_amount"] - to_sat(amount_needed) < to_sat(change_fee))
                            target["changeless_misses"] += 1
                        # otherwise pick a non-bucket utxo instead of nearest utxo
                        else:
                            # use the next largest non-bucket utxo instead
                            utxo_amount = spend_from_large_non_bucket(amount_needed, sorted_utxos, target_counts)
                else:
                    # add excess to target value if less than change fee
                    changeless = True

            if not changeless and utxo_amount >= amount_needed:
                # found solution with change
                change_amount = utxo_amount - amount_needed
                outputs_sats = []
                # check that change is larger than change fee + smallest possible target
                if to_sat(change_amount) > to_sat(change_fee) + target_counts[-1]["target_amount"]:
                    capacity, change_target = next_change_target(target_counts, to_sat(change_amount) - to_sat(change_fee))
                    if refill and capacity <= 1.0:
                        # split change to refill as many buckets as possible
                        outputs_sats = split_to_change_targets(to_sat(change_amount), change_target, target_counts, to_sat(change_fee), -1)
                    else:
                        # a single change output atleast as big as the smallest possible target + lowest spend fee
                        outputs_sats = [to_sat(change_amount - change_fee)]
                    outputs = []
                    for sats in outputs_sats:
                        assert(to_sat(to_coin(sats)) == sats)
                        outputs.append(to_coin(sats))
                        if sats < bucket_max:
                            if not is_target(sats, target_counts):
                                small_change_outputs_not_in_buckets.append(sats)
                        
                    total_fees = utxo_amount - (spend + sum(outputs))
                    assert(total_fees == tx_fee + len(outputs)*change_fee)
                    check_fees(feerate, tx_size_vbytes, total_fees, outputs, change_fee)
                    pending_txs.append(outputs)
                    total_sent_value += Decimal(spend)
                    with_change_txs.append({"utxo_amount": Decimal(utxo_amount), "feerate": feerate, "excess": satoshi_round(Decimal(utxo_amount) - amount_needed)})
                    success = True
                else:
                    # add change to target value if less than change fee + smallest possible target + lowest spend fee
                    changeless = True
            else:
                failures.append({"largest_utxo":sorted_utxos[-1], "amount_needed": amount_needed})
                success = False

            if changeless:
                pending_txs.append([])
                total_fees = tx_fee
                total_sent_value += Decimal(utxo_amount - tx_fee)
                changeless_txs.append({"spend": spend, "utxo_amount": Decimal(utxo_amount), "feerate": feerate, "excess": satoshi_round(Decimal(utxo_amount) - amount_needed)})
                success = True
            
            if success:
                sim_total_fees += total_fees
                assert(utxo_amount in utxos)
                utxos.remove(utxo_amount)
            else:
                failures[-1]["spend"] = to_coin(spend)
                failures[-1]["tx_fee"] = tx_fee
                print(failures[-1])

        # add pending change utxos
        if len(pending_txs) >= 50:
            confirmed_utxos = pending_txs.pop(0)
            for utxo in confirmed_utxos:
                utxos.append(utxo)

        count += 1
        if count % 100 == 0:
            avg_capacity = 0
            for target in target_counts: 
                avg_capacity += float(target["count"]) / (target["target_count"])
            avg_capacity = Decimal(100*avg_capacity/len(target_counts)).quantize(Decimal('0.01'))
            extra_fees = Decimal(100*sim_total_fees/min_total_fees).quantize(Decimal('0.01'))
            changeless = Decimal(100*len(changeless_txs)/(len(with_change_txs)+len(changeless_txs))).quantize(Decimal('0.01'))
            last_changeless = Decimal(100*(len(changeless_txs)-last_changeless_txs)/(len(with_change_txs)+len(changeless_txs)-last_total_txs)).quantize(Decimal('0.01'))
            print(f"{count}: utxos: {len(utxos)}, avg_capacity: {avg_capacity}%, extra_fees: {extra_fees}%, changeless: {changeless}%, changeless (last 100): {last_changeless}%")
            last_changeless_txs = len(changeless_txs)
            last_total_txs = len(with_change_txs) + len(changeless_txs)
            tmp_counts = np.unique([c['count'] for c in target_counts])
            print(f"target_counts: {tmp_counts}")

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
        if not is_target(to_sat(utxo), target_counts):
            utxos_in_buckets.append(to_sat(utxo))
        else:
            utxos_not_in_buckets.append(to_sat(utxo))
            

    for tx in pending_txs:
        for utxo in tx:
            if not is_target(to_sat(utxo), target_counts):
                utxos_in_buckets.append(utxo)
            else:
                utxos_not_in_buckets.append(utxo)

    small_utxos_not_in_bucket = []
    for utxo in utxos_not_in_buckets:
        if utxo < bucket_max:
            small_utxos_not_in_bucket.append(utxo)

    sorted_target_counts = compute_target_counts(utxos, pending_txs, utxo_targets, feerates_satoshis_per_kvbyte)
    
    results_stats["extra_fees_percent"]=(sim_total_fees/min_total_fees)-1
    results_stats["excess_percent"]=sim_total_excess/total_sent_value
    results_stats["changeless_utxos_percent"]=len(changeless_txs)/(len(with_change_txs)+len(changeless_txs))
    results_stats["bucket_refill_feerate"]=bucket_refill_feerate
    results_stats["minimum_total_fees"]=min_total_fees
    results_stats["simulated_total_fees"]=sim_total_fees
    results_stats["simple_total_fees"]=simple_total_fees
    results_stats["simulated_total_excess"]=sim_total_excess
    results_stats["utxos_count"]=len(utxos)
    results_stats["target_counts"]=sorted_target_counts
    results_stats["total_sent_value"]=total_sent_value
    results_stats["total_utxo_value"]=sum(utxos)
    results_stats["total_pending_utxo_value"]=total_pending_utxo_value
    results_stats["total_utxo_value+total_sent_value+total_fees+total_pending_utxo_value"]=sum(utxos)+total_sent_value+sim_total_fees+total_pending_utxo_value
    results_stats["total_funding_value"]=total_funding_value
    results_stats["received_utxos"]=len(received_utxos)
    results_stats["received_utxos_value"]=sum(received_utxos)
    results_stats["total_changeless"]=len(changeless_txs)
    results_stats["total_with_change"]=len(with_change_txs)
    results_stats["utxos_not_in_buckets"]=len(utxos_not_in_buckets)
    results_stats["utxos_in_buckets"]=len(utxos_in_buckets)
    results_stats["small_change_outputs_not_in_buckets"]=len(small_change_outputs_not_in_buckets)
    results_stats["small_utxos_not_in_buckets"]=len(small_utxos_not_in_bucket)
    results_stats["changeless_misses"]=changeless_misses
    results_stats["failure_count"]=len(failures)
    results_stats["failures"]=str(failures)
    results.writerow(results_stats)

    print(f"minimum_total_fees: {min_total_fees} 1.0x")
    print(f"simple_total_fees: {simple_total_fees} {Decimal(simple_total_fees/min_total_fees).quantize(Decimal('0.01'))}x")
    print(f"simulated_total_fees: {sim_total_fees} {Decimal(sim_total_fees/min_total_fees).quantize(Decimal('0.01'))}x")