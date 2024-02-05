#! /usr/bin/env python3

import argparse
import ast
import csv
from decimal import Decimal
import json
import os

import matplotlib.pyplot as plt
import numpy as np

COIN = 100000000

def to_sat(amount):
    return int(amount * COIN)


parser = argparse.ArgumentParser("Split a given utxos.csv file into a scatter plot binned into target buckets")
parser.add_argument("utxos", help="CSV file to import utxo sets from")
parser.add_argument("--bin_size", default=1000, type=int, help="sizes of bins in sats, default: 1000")
parser.add_argument("--group", default=1, type=int, help="number of ops to combine per point, default: 100")
args = parser.parse_args()

decoder = json.JSONDecoder()

xvals=[]
yvals=[]
zvals=[]
with open(args.utxos) as utxo_file:
    new_f = open(os.path.splitext(args.utxos)[0] + f"-scatter.csv", "w")
    utxoreader = csv.reader(utxo_file, delimiter=',', quotechar='"')
    next(utxoreader)

    time = 0
    for row in utxoreader:
        utxos = ast.literal_eval(row[1])
        if (((time % args.group) == 0)):
            bin_counts = {}

        for i in utxos:
            amount = float(i)
            bin = to_sat(amount/args.bin_size)
            if bin not in bin_counts:
                bin_counts[bin] = 1
            else:
                bin_counts[bin] += 1

        if ((((time+1) % args.group) == 0)):
            for point in bin_counts:
                # Column 1: Enter a label. The label shows up inside the bubble.
                # Column 2: Enter values for the X axis.
                # Column 3: Enter values for the Y axis. Each column of Y-values will show as a series of points on the chart.
                # Column 4: Enter the name of the data series. Each data series is a different color. The name shows up in the legend.
                # Column 5: Enter a number for the size of the bubble.
                for bin in bin_counts:
                    #row = ","+str(time)+","
                    #row += str(bin) + ",,"+str(bin_counts[bin])+"\n"
                    #new_f.write(row)
                    xvals.append(time)
                    yvals.append(bin*args.bin_size)
                    zvals.append(bin_counts[bin]/args.group)

        time += 1

nxvals = np.array(xvals)
nyvals = np.array(yvals)
nzvals = np.array(zvals)

nzvals1 = (10 * nzvals / nzvals.max())**2

fig, ax = plt.subplots()

ax.set_yscale('log')
ax.scatter(nxvals, nyvals, nzvals1, alpha=0.5)

ax.set_xlabel(r'time', fontsize=15)
ax.set_ylabel(r'amount (sats)', fontsize=15)
ax.set_title(args.utxos+'\nbin_size='+str(args.bin_size)+', group='+str(args.group))

ax.grid(True)
fig.tight_layout()

plt.show()
