#! /usr/bin/env python3

import argparse
import json

from decimal import Decimal, ROUND_DOWN, ROUND_UP
from random import random, randrange

COIN = 100000000
SATOSHI = Decimal(0.00000001)
KILO = 1000

def satoshi_round(amount):
    return Decimal(amount).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

def to_sat(amount):
    return int(amount * COIN)

def to_coin(amount):
    return satoshi_round(amount / COIN)

def random_bucket(utxotargets, max_utxos):
    index = randrange(0, max_utxos)
    count = 0
    for bucket in utxotargets:
        count += bucket['target_utxo_count']
        if (index < count):
            return bucket
        
def get_amount(bucket, receive_chance, long_term_rate):
    if random() < receive_chance:
        # we can not receive value less than what it would cost to spend it at the long term rate
        min_receive = int(Decimal(COIN * Decimal(long_term_rate) * 43 / Decimal(1000.0)).quantize(1, ROUND_UP))
        # receive funds from the close of random bucket with value between min_receive and the funding amount of the bucket
        amount = randrange(min_receive, bucket['start_satoshis'])
    else:
        # send funds from the funding of a random bucket with value beween the min/max satoshis for that bucket
        amount = -1*bucket['start_satoshis']
    return to_coin(amount)

def read_feerate(fin):
    line = fin.readline()
    if (line != ''):
        return Decimal(line.strip().split(',')[-1])
    return False

def write_line(fout, amount, feerate):
    fout.write(f"{amount:.8f},{feerate:.8f}\n")

parser = argparse.ArgumentParser(description="Generates a simulation scenario from a utxo targets json file.")
parser.add_argument("targets", help="JSON file to import utxo targets from")
parser.add_argument("feerates", help="Fee rates csv file (btc), note: uses last value on each line.")
parser.add_argument("filename", help="File to output to")
parser.add_argument("--receive_chance", default=0.01, type=float, help="chance of receiving vs sending a payment per time point, default: 0.01 (ie. 1:100)")
parser.add_argument("--long_term_rate", default=0.0003, type=float, help="long term average feerate, default: 0.0030000 BTC/kvb")

args = parser.parse_args()

with open(args.targets, "r") as ftargets:
    decoder = json.JSONDecoder()
    text = ftargets.read()
    utxotargets = decoder.decode(text)['buckets']
    max_utxos = 0
    receive = random() 
    for bucket in utxotargets:
        max_utxos += bucket['target_utxo_count']

with open(args.filename, "w") as fout:
    with open(args.feerates, "r") as ffees:
        while(1):
            feerate = read_feerate(ffees)
            if (feerate == False):
                break
            bucket = random_bucket(utxotargets, max_utxos)
            amount = get_amount(bucket, args.receive_chance, args.long_term_rate)             
            write_line(fout, amount, feerate)
