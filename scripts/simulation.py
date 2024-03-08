#! /usr/bin/env python3
import csv
import git
import json
import logging
import os
import struct
import uuid

from authproxy import JSONRPCException
from bcc import BPF, USDT
from bisect import bisect
from collections import defaultdict
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_UP
from framework import Simulation
from random import random, randrange
from statistics import mean, stdev

from random import randrange

COIN = 100000000
KILO = 1000

def satoshi_round(amount):
    return Decimal(amount).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)

def to_sat(amount):
    return int(amount * COIN)

def to_coin(amount):
    return satoshi_round(amount / COIN)

program = """
#include <uapi/linux/ptrace.h>

#define WALLET_NAME_LENGTH 16
#define ALGO_NAME_LENGTH 16

struct event_data
{
    u8 type;
    char wallet_name[WALLET_NAME_LENGTH];

    // selected coins event
    char algo[ALGO_NAME_LENGTH];
    s64 target;
    s64 waste;
    s64 selected_value;

    // create tx event
    u8 success;
    s64 fee;
    s32 change_pos;

    // aps create tx event
    u8 use_aps;
};

BPF_QUEUE(coin_selection_events, struct event_data, 1024);

int trace_selected_coins(struct pt_regs *ctx) {
    struct event_data data;
    __builtin_memset(&data, 0, sizeof(data));
    data.type = 1;
    bpf_usdt_readarg_p(1, ctx, &data.wallet_name, WALLET_NAME_LENGTH);
    bpf_usdt_readarg_p(2, ctx, &data.algo, ALGO_NAME_LENGTH);
    bpf_usdt_readarg(3, ctx, &data.target);
    bpf_usdt_readarg(4, ctx, &data.waste);
    bpf_usdt_readarg(5, ctx, &data.selected_value);
    coin_selection_events.push(&data, 0);
    return 0;
}

int trace_normal_create_tx(struct pt_regs *ctx) {
    struct event_data data;
    __builtin_memset(&data, 0, sizeof(data));
    data.type = 2;
    bpf_usdt_readarg_p(1, ctx, &data.wallet_name, WALLET_NAME_LENGTH);
    bpf_usdt_readarg(2, ctx, &data.success);
    bpf_usdt_readarg(3, ctx, &data.fee);
    bpf_usdt_readarg(4, ctx, &data.change_pos);
    coin_selection_events.push(&data, 0);
    return 0;
}

int trace_attempt_aps(struct pt_regs *ctx) {
    struct event_data data;
    __builtin_memset(&data, 0, sizeof(data));
    data.type = 3;
    bpf_usdt_readarg_p(1, ctx, &data.wallet_name, WALLET_NAME_LENGTH);
    coin_selection_events.push(&data, 0);
    return 0;
}

int trace_aps_create_tx(struct pt_regs *ctx) {
    struct event_data data;
    __builtin_memset(&data, 0, sizeof(data));
    data.type = 4;
    bpf_usdt_readarg_p(1, ctx, &data.wallet_name, WALLET_NAME_LENGTH);
    bpf_usdt_readarg(2, ctx, &data.use_aps);
    bpf_usdt_readarg(3, ctx, &data.success);
    bpf_usdt_readarg(4, ctx, &data.fee);
    bpf_usdt_readarg(5, ctx, &data.change_pos);
    coin_selection_events.push(&data, 0);
    return 0;
}
"""


def ser_compact_size(l):
    r = b""
    if l < 253:
        r = struct.pack("B", l)
    elif l < 0x10000:
        r = struct.pack("<BH", 253, l)
    elif l < 0x100000000:
        r = struct.pack("<BI", 254, l)
    else:
        r = struct.pack("<BQ", 255, l)
    return r

def compute_target_funding(utxo_targets):
    target_funding = 0
    for bucket in utxo_targets:
        target_funding += bucket['end_satoshis']
    return target_funding

def target_from_amount(amount, utxo_targets): 
    for bucket in range(len(utxo_targets)):
        if amount >= utxo_targets[bucket]['start_satoshis'] and amount <= utxo_targets[bucket]['end_satoshis']:
            return bucket
    return -1
        
def next_change_target(utxo_targets, target_counts, change_fee):
    min_bucket = 0
    min_capacity = None
    for i in range(len(utxo_targets)):
        capacity = float(target_counts[i]) / utxo_targets[i]['target_utxo_count']
        if min_capacity is None or capacity < min_capacity:
            min_bucket = i
            min_capacity = capacity
    amount = randrange(utxo_targets[min_bucket]['start_satoshis'], utxo_targets[min_bucket]['end_satoshis'] - change_fee)
    return min_bucket, min_capacity, amount
    
# split change output to replentish utxo_targets
def split_to_change_targets(change_amount, change_target, utxo_targets, target_counts, change_fee):
    # compute initial change target
    change_outputs = [change_target]
    change_amount -= change_target
    bucket = target_from_amount(change_target, utxo_targets)
    target_counts[bucket] += 1

    while (True):
        # compute next target amount
        bucket, _, target_amount = next_change_target(utxo_targets, target_counts, change_fee)
        if (target_counts[bucket] >= utxo_targets[bucket]['target_utxo_count']):
            break
        if (change_amount < change_fee + target_amount):
            break
        change_amount -= change_fee + target_amount
        target_counts[bucket] += 1
        # add initial change output
        change_outputs.append(target_amount)

    # remaining change goes into the last change output 
    change_outputs[-1] += change_amount

    return change_outputs

class CoinSelectionSimulation(Simulation):
    def set_sim_params(self):
        self.num_nodes = 1
        self.extra_args = [["-dustrelayfee=0", "-maxtxfee=1"]]
        self.output_types = ["bech32m", "bech32", "p2sh-segwit", "legacy"]

    def log_sim_results(self, res_file, csvw):
        getcontext().prec = 12
        # Find change stats
        change_vals = sorted(self.change_vals)
        min_change = Decimal(change_vals[0]) if len(self.change_vals) > 0 else 0
        max_change = Decimal(change_vals[-1]) if len(self.change_vals) > 0 else 0
        mean_change = (
            Decimal(mean(change_vals)) * Decimal(1) if len(self.change_vals) > 0 else 0
        )
        stdev_change = (
            Decimal(stdev(change_vals)) * Decimal(1) if len(self.change_vals) > 0 else 0
        )

        # Remaining utxos and fee stats
        remaining_utxos = self.tester.listunspent()
        cost_to_empty = (
            Decimal(len(remaining_utxos)) * Decimal(68) * Decimal(0.0001) / Decimal(1000)
        )
        total_cost = self.total_fees + cost_to_empty
        mean_fees = (
            Decimal(self.total_fees) / Decimal(self.withdraws)
            if self.withdraws > 0
            else 0
        )

        # input stats
        input_sizes = sorted(self.input_sizes)
        min_input_size = Decimal(input_sizes[0]) if len(self.input_sizes) > 0 else 0
        max_input_size = Decimal(input_sizes[-1]) if len(self.input_sizes) > 0 else 0
        mean_input_size = (
            (Decimal(mean(input_sizes)) * Decimal(1))
            if len(self.input_sizes) > 0
            else 0
        )
        stdev_input_size = (
            (Decimal(stdev(input_sizes)) * Decimal(1))
            if len(self.input_sizes) > 0
            else 0
        )

        # Output counts
        output_sizes = sorted(self.output_sizes)
        max_output_size = Decimal(output_sizes[-1]) if len(output_sizes) > 0 else 0
        mean_output_size = (
            (Decimal(mean(output_sizes)) * Decimal(1))
            if len(output_sizes) > 0
            else 0
        )
        stdev_output_size = (
            (Decimal(stdev(output_sizes)) * Decimal(1))
            if len(output_sizes) > 0
            else 0
        )

        # UTXO stats
        mean_utxo_set_size = (
            (Decimal(mean(self.utxo_set_sizes)) * Decimal(1))
            if len(self.utxo_set_sizes) > 0
            else 0
        )

        # No change counts
        no_change_str = ""
        no_change_total = 0
        for algo, c in self.no_change.items():
            no_change_total += c
            no_change_str += f"{algo}: **{c}** ; "
        no_change_str += f"Total: **{no_change_total}**"

        # Usage counts
        usage_str = ""
        for algo, c in self.algo_counts.items():
            usage_str += f"{algo}: **{c}** ; "
        usage_str = usage_str[:-3]

        unlocked_balance = 0
        for utxo in remaining_utxos:
            unlocked_balance += utxo['amount']

        result = [
            self.scenario_name,
            str(unlocked_balance),
            str(mean_utxo_set_size),
            str(len(remaining_utxos)),
            str(self.count_received),
            str(self.count_sent),
            str(self.withdraws),
            str(self.unec_utxos),
            str(len(self.change_vals)),
            no_change_str,
            str(min_change),
            str(max_change),
            str(mean_change),
            str(stdev_change),
            str(self.total_fees),
            str(mean_fees),
            str(cost_to_empty),
            str(total_cost),
            str(min_input_size),
            str(max_input_size),
            str(mean_input_size),
            str(stdev_input_size),
            str(max_output_size),
            str(mean_output_size),
            str(stdev_output_size),
            usage_str,
        ]
        result_str = f"| {' | '.join(result)} |"
        res_file.write(f"{result_str}\n")
        res_file.flush()
        self.log.debug(result_str)
        csvw.writerow(result)
        return result_str
    
    def compute_target_counts(self, utxo_targets):
        all_unspent_utxos = self.tester.listunspent(0)
        target_counts = [0]*len(utxo_targets)
        for utxo in all_unspent_utxos:
            bucket = target_from_amount(to_sat(utxo['amount']), utxo_targets)
            if bucket > -1:
                target_counts[bucket] += 1
        return target_counts
    
    def log_mempool_txs(self, txids):
        for txid in txids:
            try:
                results = self.funder.getmempoolentry(txid)
                self.log.warning(
                    f"txid: {txid}, depends: {results['depends']}, spentby: {results['spentby']}"
                )
            except JSONRPCException as e:
                self.log.warning(f"txid not found in mempool: {txid}")

    def run(self):
        # Get Git commit
        repo = git.Repo(self.config["environment"]["SRCDIR"])
        commit = repo.commit("HEAD")
        commit_hash = commit.hexsha
        branch = repo.active_branch.name
        if self.options.label is None:
            self.log.info(f"Based on branch {branch}({commit_hash})")
        else:
            label = self.options.label
            self.log.info(f"Based on branch: {branch} ({commit_hash}), label: {label}")

        # Get a unique id
        unique_id = uuid.uuid4().hex
        self.log.info(f"This simulation's Unique ID: {unique_id}")

        # Make an output folder
        if self.options.label is None:
            results_dir = os.path.join(
                self.options.resultsdir, f"{branch}-{commit_hash}", f"sim_{unique_id}"
            )
        else:
            results_dir = os.path.join(
                self.options.resultsdir,
                f"{branch}-{commit_hash}-{label}-",
                f"sim_{unique_id}",
            )
        os.makedirs(results_dir, exist_ok=True)

        # Setup debug logging
        debug_log_handler = logging.FileHandler(
            os.path.join(results_dir, "sim_debug.log")
        )
        debug_log_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d000Z %(name)s (%(levelname)s): %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        debug_log_handler.setFormatter(formatter)
        self.log.addHandler(debug_log_handler)

        # Decimal precision
        getcontext().prec = 12

        # Make two wallets
        self.nodes[0].createwallet(wallet_name="funder", descriptors=True)
        self.nodes[0].createwallet(wallet_name="tester", descriptors=True)
        self.funder = self.nodes[0].get_wallet_rpc("funder")
        self.tester = self.nodes[0].get_wallet_rpc("tester")

        # Check that there's no UTXO on the wallets
        assert len(self.funder.listunspent()) == 0
        assert len(self.tester.listunspent()) == 0

        self.log.info("Mining blocks for node0 to be able to send enough coins")

        gen_addr = self.funder.getnewaddress()
        self.funder.generatetoaddress(600, gen_addr)  # > 14,000 BTC
        withdraw_addresses = {
            output: self.funder.getnewaddress(address_type=output)
            for output in self.output_types
        }

        # set this as the default. if weights are provided by the user, we will update this when creating the psbt
        withdraw_address = withdraw_addresses["bech32m"]
        if self.options.scenario:
            self.scenario_name = os.path.splitext(os.path.basename(self.options.scenario))[0]

            def get_scenario_data(file):
                for line in file:
                    val_str, fee_str = line.rstrip().lstrip().split(",")
                    yield val_str, fee_str

            scenario_files = [open(self.options.scenario, "r")]
            scenario_data = get_scenario_data(scenario_files[0])
        elif self.options.payments and self.options.feerates:
            self.scenario_name = f"{os.path.splitext(os.path.basename(self.options.payments))[0]}_{os.path.splitext(os.path.basename(self.options.feerates))[0]}"

            def cycle(file):
                while True:
                    for line in file:
                        yield line
                    file.seek(0)

            scenario_files = [
                open(self.options.payments, "r"),
                open(self.options.feerates, "r"),
            ]
            scenario_data = zip(scenario_files[0], cycle(scenario_files[1]))

        self.scenario_path = self.options.scenario

        utxo_targets = {}
        bucket_refill_feerate = 0
        if self.options.utxo_targets:
            ftargets = open(self.options.utxo_targets)
            decoder = json.JSONDecoder()
            text = ftargets.read()
            json_dec = decoder.decode(text)
            utxo_targets = json_dec['buckets']
            bucket_refill_feerate = json_dec['bucket_refill_feerate']

        targets_python = "utxotargetsfile" not in self.options.extra_configs and len(utxo_targets) > 0

        fields = [
            "Scenario File",
            "Current Balance",
            "Mean #UTXO",
            "Current #UTXO",
            "#Deposits",
            "#Inputs Spent",
            "#Withdraws",
            "#Uneconomical outputs spent",
            "#Change Created",
            "#Changeless",
            "Min Change Value",
            "Max Change Value",
            "Mean Change Value",
            "Std. Dev. of Change Value",
            "Total Fees",
            "Mean Fees per Withdraw",
            "Cost to Empty (10 sat/vB)",
            "Total Cost",
            "Min Input Size",
            "Max Input Size",
            "Mean Input Size",
            "Std. Dev. of Input Size",
            "Max Output Size",
            "Mean Output Size",
            "Std. Dev. of Output Size",
            "Usage",
        ]
        header = f"| {' | '.join(fields)} |"

        # Connect tracepoints
        bitcoind_with_usdts = USDT(pid=self.nodes[0].process.pid)
        bitcoind_with_usdts.enable_probe(
            probe="selected_coins", fn_name="trace_selected_coins"
        )
        bitcoind_with_usdts.enable_probe(
            probe="normal_create_tx_internal", fn_name="trace_normal_create_tx"
        )
        bitcoind_with_usdts.enable_probe(
            probe="attempting_aps_create_tx", fn_name="trace_attempt_aps"
        )
        bitcoind_with_usdts.enable_probe(
            probe="aps_create_tx_internal", fn_name="trace_aps_create_tx"
        )
        bpf = BPF(text=program, usdt_contexts=[bitcoind_with_usdts])

        self.log.info(f"Simulating using scenario: {self.scenario_name}")
        self.log.info(f"Options: {self.options}")
        self.total_fees = Decimal()
        self.ops = 0
        self.count_sent = 0
        self.change_vals = []
        self.no_change = defaultdict(int)
        self.withdraws = 0
        self.input_sizes = []
        self.output_sizes = []
        self.utxo_set_sizes = []
        self.count_change = 0
        self.count_received = 0
        self.unec_utxos = 0
        self.algo_counts = defaultdict(int)
        self.pending_txs = []
        with open(
            os.path.join(results_dir, "full_results.csv"), "a+"
        ) as full_res, open(
            os.path.join(results_dir, "results.txt"), "a+"
        ) as res, open(
            os.path.join(results_dir, "results.csv"), "a+"
        ) as csv_res, open(
            os.path.join(results_dir, "utxos.csv"), "a+"
        ) as utxos_res, open(
            os.path.join(results_dir, "inputs.csv"), "a+"
        ) as inputs_res:

            dw = csv.DictWriter(
                full_res,
                [
                    "id",
                    "amount",
                    "fees",
                    "target_feerate",
                    "real_feerate",
                    "algo",
                    "num_inputs",
                    "negative_ev",
                    "num_outputs",
                    "change_amount",
                    "before_num_utxos",
                    "after_num_utxos",
                    "waste",
                ],
            )
            dw.writeheader()
            utxos_dw = csv.DictWriter(utxos_res, ["id", "utxo_amounts"])
            utxos_dw.writeheader()
            inputs_dw = csv.DictWriter(inputs_res, ["id", "input_amounts"])
            inputs_dw.writeheader()
            sum_csvw = csv.writer(csv_res)
            sum_csvw.writerow(fields)

            res.write(
                f"----BEGIN SIMULATION RESULTS----\nScenario: {self.scenario_name}\n{header}\n"
            )
            res.flush()
            for val_str, fee_str in scenario_data:

                confirm_transactions = []
                value = Decimal(val_str.strip())
                feerate = Decimal(fee_str.strip())
                # cost to add a taproot output
                change_fee = int(Decimal(COIN * feerate * Decimal(43) / Decimal(1000.0)).quantize(1, ROUND_UP))

                if self.options.ops and self.ops > self.options.ops:
                    break
                if self.ops % 500 == 0:
                    self.log.info(f"{self.ops} operations performed so far")
                    self.log_sim_results(res, sum_csvw)

                # compute current state of target utxo buckets
                before_utxos = self.tester.listunspent()
                target_counts = self.compute_target_counts(utxo_targets)

                # add funding when our largest confirmed utxo is from a bucket
                max_utxo = None
                for utxo in before_utxos:
                    if max_utxo is None or utxo['amount'] > max_utxo['amount']:
                        max_utxo = utxo

                if max_utxo is None:
                    refill_funding = True
                else:
                    if len(utxo_targets) > 0:
                        # add more funding when our largest utxo is in one of our target buckets
                        bucket = target_from_amount(to_sat(max_utxo['amount']), utxo_targets)
                        refill_funding = bucket != -1
                    else:
                        refill_funding = max_utxo['amount'] < 0.001

                if refill_funding:
                    try:
                        if len(utxo_targets) > 0:
                            # add funding and split it based on targets
                            _, _, change_target = next_change_target(utxo_targets, target_counts, 0)
                            outputs = split_to_change_targets(to_sat(self.options.payment_amount), change_target, utxo_targets, target_counts, change_fee)
                            change_amounts = {self.tester.getnewaddress(address_type='bech32m'): to_coin(change_out) for change_out in outputs}
                            txid = self.funder.send(outputs=change_amounts, options={"subtract_fee_from_outputs":[*range(0,len(outputs))]})['txid']
                            self.log.debug(
                                f"Op {self.ops} UTXOs below minimum, added deposit of {self.options.payment_amount} BTC with {len(outputs)} outputs"
                            )
                            confirm_transactions.append(txid)
                            before_utxos = self.tester.listunspent()
                            target_counts = self.compute_target_counts(utxo_targets)
                        else:
                            # add funding and split it evenly
                            value = to_coin(to_sat(self.options.payment_amount)/255)
                            change_amounts = {self.tester.getnewaddress(address_type='bech32m'): value for _ in range(0,255)}
                            txid = self.funder.send(outputs=change_amounts, options={"subtract_fee_from_outputs":[*range(0,255)]})['txid']
                            confirm_transactions.append(txid)
                            before_utxos = self.tester.listunspent()
                            self.log.debug(
                                f"Op {self.ops} UTXOs below minimum, added deposit of {self.options.payment_amount} BTC with 255 outputs"
                            )

                    except JSONRPCException as e:
                        self.log.warning(
                            f"Failure on op {self.ops} with funder sending {self.options.payment_amount} with error {str(e)}"
                        )
                
                # Make deposit or withdrawal
                if self.options.weights:
                    # choose a random address type based on the weights provided by the user
                    i = bisect(self.options.weights, random() * 100)
                    withdraw_address = withdraw_addresses[self.output_types[i]]

                if value > 0:
                    try:
                        # deposit
                        txid = self.funder.send(
                            outputs=[{self.tester.getnewaddress(): value}], 
                            options={"change_address": withdraw_address}
                        )['txid']
                        confirm_transactions.append(txid)
                        self.count_received += 1
                        self.log.debug(
                            f"Op {self.ops} Received {self.count_received}th deposit of {value} BTC"
                        )
                    except JSONRPCException as e:
                        self.log.warning(
                            f"Failure on op {self.ops} with funder sending {value} with error {str(e)}"
                        )
                    # Make sure all tracepoint events are consumed
                    try:
                        while True:
                            bpf["coin_selection_events"].pop()
                    except KeyError:
                        pass
                if value < 0:
                    try:
                        payment_stats = {"id": self.withdraws}
                        # Before listunspent
                        payment_stats["before_num_utxos"] = len(before_utxos)
                        utxo_amounts = [str(u["amount"]) for u in before_utxos]
                        utxos_dw.writerow(
                            {"id": self.withdraws, "utxo_amounts": utxo_amounts}
                        )
                        # Prepare withdraw
                        value = value * -1
                        payment_stats["amount"] = value
                        payment_stats["target_feerate"] = feerate
                        # use the bech32 withdraw address by default
                        # if weights are provided, then choose an address type based on the provided distribution
                        spend_inputs = []
                        spend_outputs=[{withdraw_address: value}]
                        funding_options= {
                            "feeRate": feerate,
                            "lockUnspents": True,
                            "changePosition": 1,
                            "add_inputs": True,
                            "minconf": 1
                        }
                        if self.options.disable_algos == True:
                            funding_options["disable_algos"] = ["knapsack","srd"]
                        if targets_python:
                            _, min_capacity, change_target = next_change_target(utxo_targets, target_counts, change_fee)
                            if min_capacity < 1.0 and targets_python:
                                funding_options["change_target"] = to_coin(change_target)
                            # refill buckets if fee rate is low or a bucket is almost empty
                            refill = feerate < to_coin(bucket_refill_feerate)
                            if max_utxo is not None and ((min_capacity < 0.3) or (min_capacity < 0.7 and refill)):
                                if (min_capacity < 0.3):
                                    self.log.debug(
                                        f"Op {self.ops} Pre-emptively add input to refill targets from UTXO of amount {max_utxo['amount']} (capacity < 30%)."
                                    )
                                else:
                                    self.log.debug(
                                        f"Op {self.ops} Pre-emptively add input to refill targets with UTXO of amount {max_utxo['amount']} (capacity < 70% and feerate {feerate} < bucket_refill_feerate {to_coin(bucket_refill_feerate)}."
                                    )

                                spend_inputs = [{
                                    "txid": max_utxo['txid'],
                                    "vout": max_utxo['vout']
                                }]
                        
                        psbt = self.tester.walletcreatefundedpsbt(
                            inputs = spend_inputs,
                            outputs= spend_outputs,
                            options= funding_options
                        )["psbt"]

                        dec = self.tester.decodepsbt(psbt)
                        if len(dec['tx']['vout']) > 1 and targets_python:
                            change_amount = to_sat(dec['tx']['vout'][1]['value'])
                            change_outputs = split_to_change_targets(change_amount, change_target, utxo_targets, target_counts, change_fee)
                            # transaction modified to add change outputs
                            input = []
                            for tx_in in dec['tx']['vin']:
                                input.append({"txid": tx_in['txid'], 'vout': tx_in['vout']})
                            for change_out in change_outputs:
                                spend_outputs += [{self.tester.getnewaddress(address_type='bech32m'): to_coin(change_out)}]
                            # create updated psbt with new outputs
                            psbt = self.tester.createpsbt(input, spend_outputs)

                        psbt = self.tester.walletprocesspsbt(psbt)["psbt"]
                        # Send the tx
                        psbt = self.tester.finalizepsbt(psbt, False)["psbt"]
                        tx = self.tester.finalizepsbt(psbt)["hex"]
                        dec = self.tester.decodepsbt(psbt)
                        
                        # decode txid to get txid and vbytes size
                        dec_tx = self.tester.decoderawtransaction(tx)

                        if len(dec_tx['vout']) > 1:
                            self.log.debug(
                                f"Op {self.ops} Opportunistically refill targets with {len(dec_tx['vout']) - 1} change outputs."
                            )

                        # add transaction to the mempool
                        self.tester.sendrawtransaction(tx)

                        # delay confirmation of tx unless refilling with a funding tx
                        if (self.options.delay_confirmation == 0 or refill_funding):
                            confirm_transactions.append(dec_tx['txid'])
                        else:
                            self.pending_txs.append(dec_tx)
                            if (len(self.pending_txs) > self.options.delay_confirmation):
                                pending_tx = self.pending_txs.pop(0)
                                confirm_transactions.append(pending_tx['txid'])

                        mempool = self.tester.getrawmempool(verbose=True)
                        if dec_tx['txid'] not in mempool:
                            self.log.warning(
                                f"Failure on op {self.ops} txid {dec_tx['txid']} submitted but not added to mempool."
                            )
                        for x in confirm_transactions:
                            if x not in mempool:
                                self.log.warning(
                                    f"Failure on op {self.ops} txid {x} confirmed but not found in mempool."
                                )
                                confirm_transactions = []
                                break

                        pending_txs1 = []
                        for x in self.pending_txs:
                            pending_txs1.append(x['txid'])
                        for x in confirm_transactions:
                            pending_txs1.append(x)
                        for x in mempool:
                            if x not in pending_txs1:
                                print(x)
                                    
                        # Get data from the tracepoints
                        algo = None
                        change_pos = None
                        waste = None
                        try:
                            is_aps = False
                            sc_events = []
                            while True:
                                event = bpf["coin_selection_events"].pop()
                                if b"tester" not in event.wallet_name:
                                    continue
                                if event.type == 1:
                                    if not is_aps:
                                        algo = event.algo.decode()
                                        waste = event.waste
                                    sc_events.append(event)
                                elif event.type == 2:
                                    assert event.success == 1
                                    if not is_aps and event.change_pos != -1:
                                        change_pos = event.change_pos
                                elif event.type == 3:
                                    is_aps = True
                                elif event.type == 4:
                                    assert is_aps
                                    if event.use_aps == 1:
                                        assert len(sc_events) == 2
                                        algo = sc_events[1].algo.decode()
                                        waste = sc_events[1].waste
                                        change_pos = event.change_pos
                        except KeyError:
                            pass
                        assert algo is not None
                        assert waste is not None
                        payment_stats["algo"] = algo
                        payment_stats["waste"] = waste
                        self.algo_counts[algo] += 1
                        # Get negative EV UTXOs
                        payment_stats["negative_ev"] = 0
                        input_amounts = []
                        for in_idx, inp in enumerate(dec["inputs"]):
                            inp_size = (
                                4 + 36 + 4
                            )  # prev txid, output index, sequence are all fixed size
                            ev = 0
                            if "final_scriptSig" in inp:
                                scriptsig_len = len(inp["final_scriptSig"])
                                inp_size += scriptsig_len + len(
                                    ser_compact_size(scriptsig_len)
                                )
                            else:
                                inp_size += 1
                            if "final_scriptWitness" in inp:
                                witness_len = len(inp["final_scriptWitness"])
                                inp_size += witness_len / 4
                            inp_fee = feerate * (Decimal(inp_size) / Decimal(1000.0))
                            if "witness_utxo" in inp:
                                utxo = inp["witness_utxo"]
                                input_amounts.append(str(inp["witness_utxo"]["amount"]))
                                ev = inp["witness_utxo"]["amount"] - inp_fee
                            else:
                                assert "non_witness_utxo" in inp
                                out_index = dec["tx"]["vin"][in_idx]["vout"]
                                utxo = inp["non_witness_utxo"]["vout"][out_index]
                                input_amounts.append(str(utxo["value"]))
                                ev = utxo["value"] - inp_fee
                            if ev <= 0:
                                self.unec_utxos += 1
                                payment_stats["negative_ev"] += 1

                        inputs_dw.writerow(
                            {"id": self.withdraws, "input_amounts": input_amounts}
                        )
                        # Get fee info
                        fee = dec['fee']
                        self.total_fees += fee
                        payment_stats["fees"] = fee
                        # Get real feerate
                        payment_stats["real_feerate"] = fee / dec_tx["vsize"] * 1000
                        # Spent utxo counts and input info
                        num_in = len(dec["inputs"])
                        num_out = len(dec["outputs"])
                        self.count_sent += num_in
                        self.input_sizes.append(num_in)
                        self.output_sizes.append(len(dec["outputs"]))
                        payment_stats["num_inputs"] = num_in
                        payment_stats["num_outputs"] = num_out
                        # Change info
                        payment_stats["change_amount"] = None
                        if change_pos is not None and change_pos != -1:
                            assert len(dec["tx"]["vout"]) >= 2
                            change_out = dec["tx"]["vout"][change_pos]
                            payment_stats["change_amount"] = change_out["value"]
                            self.change_vals.append(change_out["value"])
                            self.count_change += num_out-1
                        else:
                            assert len(dec["tx"]["vout"]) == 1
                            self.no_change[algo] += 1
                        # After listunspent
                        payment_stats["after_num_utxos"] = len(
                            self.tester.listunspent(0)
                        )
                        dw.writerow(payment_stats)
                        self.log.debug(
                            f"Op {self.ops} Sent {self.withdraws}th withdraw of {value} BTC using {num_in} inputs and {num_out} outputs with fee {fee} ({feerate} BTC/kvB) and algo {algo}"
                        )
                        self.withdraws += 1

                    except JSONRPCException as e:
                        # Make sure all tracepoint events are consumed
                        try:
                            while True:
                                bpf["coin_selection_events"].pop()
                        except KeyError:
                            pass
                        self.log.warning(
                            f"Failure on op {self.ops} with tester sending {value} with error {str(e)}"
                        )

                self.utxo_set_sizes.append(len(self.tester.listunspent(0)))
                try:
                    self.funder.generateblock(output=gen_addr, transactions=confirm_transactions)
                except JSONRPCException as e:
                    self.log.warning(
                        f"Failure on op {self.ops} with funder calling generateblock and confirming transactions: {confirm_transactions} with error {str(e)}"
                    )
                    
                self.ops += 1

            for f in scenario_files:
                f.close()

            final_result = self.log_sim_results(res, sum_csvw)
            res.write("----END SIMULATION RESULTS----\n\n\n")
            res.flush()
            self.log.info(header)
            self.log.info(final_result)


if __name__ == "__main__":
    CoinSelectionSimulation().main()
