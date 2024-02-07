#! /usr/bin/env python3

import argparse
import json

from decimal import Decimal, ROUND_DOWN
from random import randrange

COIN = 100000000
SATOSHI = Decimal(0.00000001)
KILO = 1000

def satoshi_round(amount):
    return Decimal(amount).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

def to_sat(amount):
    return int(amount * COIN)

def to_coin(amount):
    return satoshi_round(amount / COIN)

def random_send_sats(utxotargets, max_utxos):
    index = randrange(0, max_utxos)
    count = 0
    for bucket in utxotargets:
        count += bucket['target_utxo_count']
        if (index < count):
            return -1*bucket['start_satoshis']
        
def get_amount(utxotargets, max_utxos, receive_chance, receive_min_sats, receive_max_sats):
    if (randrange(0, 1000000) < receive_chance*1000000):
        amount = randrange(receive_min_sats, receive_max_sats)
    else:
        amount = random_send_sats(utxotargets, max_utxos)
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
parser.add_argument("--receive_chance", default=0.001, type=float, help="chance of receiving vs sending a payment per time point, default: 0.001 (ie. 1:1000)")
parser.add_argument("--receive_min", default=1000, type=int, help="minimium receive amount (sats), default: 1000")
parser.add_argument("--receive_max", default=10000000, type=int, help="maximum receive amount (sats), default: 10000000")
parser.add_argument("--average_tx_size", default=200, type=int, help="estimaged average tx size in vBytes, default: 200 vB")

args = parser.parse_args()

with open(args.targets, "r") as ftargets:
    decoder = json.JSONDecoder()
    text = ftargets.read()
    utxotargets = decoder.decode(text)['buckets']
    max_utxos = 0
    for bucket in utxotargets:
        max_utxos += bucket['target_utxo_count']
    funding_balance = Decimal(0)

with open(args.filename, "w") as fout:
    with open(args.feerates, "r") as ffees:
        while(1):
            feerate = read_feerate(ffees)
            if (feerate == False):
                break
            amount = get_amount(utxotargets, max_utxos, args.receive_chance, args.receive_min, args.receive_max)             
            write_line(fout, amount, feerate)
            #  deduct amount sent, or credit amount received
            funding_balance += to_sat(amount)
            # deduct estimated fees from running funding balance
            funding_balance -= to_sat(Decimal(args.average_tx_size)/1000*feerate)
