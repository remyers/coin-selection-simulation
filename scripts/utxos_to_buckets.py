#! /usr/bin/env python3

import argparse
import ast
import csv
import json
import os

COIN = 100000000

def to_sat(amount):
    return int(amount * COIN)

def amount_to_bucket(amount, utxotargets):
    index = 0
    for target in utxotargets:
        if amount < target['start_satoshis']:
            return index
        if amount >= target['start_satoshis'] and amount < target['end_satoshis']:
            return index+1
        else:
            index+=2

parser = argparse.ArgumentParser("Split a given utxos.csv file into a scatter plot binned into target buckets")
parser.add_argument("utxos", help="CSV file to import utxo sets from")
parser.add_argument("targets", help="targets json file")
args = parser.parse_args()

decoder = json.JSONDecoder()

with open(args.utxos) as utxo_file:
    new_f = open(os.path.splitext(args.targets)[0] + f"-utxos.csv", "w")
    utxoreader = csv.reader(utxo_file, delimiter=',', quotechar='"')
    next(utxoreader)

    max_bin = 0
    with open(args.targets, "r") as ftargets:
        decoder = json.JSONDecoder()
        text = ftargets.read()
        utxotargets = decoder.decode(text)['buckets']
        max_bin = len(utxotargets)*2
        header = "time"
        start = 0
        for bin in range(max_bin):
            if bin % 2 == 0:
                end = utxotargets[int(bin/2)]['start_satoshis']
                target = 0
            else:
                start = utxotargets[int(bin/2)]['start_satoshis']
                end = utxotargets[int(bin/2)]['end_satoshis']
                target = utxotargets[int(bin/2)]['target_utxo_count']
            header += ", " + str(start) + "-" + str(end) + " ["+str(target)+"]"
            start = end
        header += ", " + str(start)+"-max [0]\n"
        max_bin+=1

    new_f.write(header)

    time = 0
    for row in utxoreader:
        utxos = ast.literal_eval(row[1])
        bin_counts = {}

        for i in utxos:
            amount = float(i)
            bin = amount_to_bucket(to_sat(amount), utxotargets)
            if bin not in bin_counts:
                bin_counts[bin] = 1
            else:
                bin_counts[bin] += 1

        out = str(time)
        for bin in range(max_bin):
            if (bin in bin_counts): 
                if bin % 2 == 1:
                    out += ", " + str(float(bin_counts[bin]))
                else:
                    out += ", " + str(float(bin_counts[bin]))
            else:
                out += ", 0"
        new_f.write(out+"\n")                    

        time += 1

