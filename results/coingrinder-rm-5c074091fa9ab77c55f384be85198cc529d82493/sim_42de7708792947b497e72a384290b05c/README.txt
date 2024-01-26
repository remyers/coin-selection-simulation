Command line:

/usr/bin/env sudo -E /bin/python3 /home/remyers/.vscode/extensions/ms-python.python-2023.22.1/pythonFiles/lib/python/debugpy/adapter/../../debugpy/launcher 36861 -- scripts/simulation.py --extra_config utxotargetsfile=/home/remyers/github/coin-selection-simulation/utxotargets.json --tmpdir /mnt/tmp/bitcoin_coin_sel_sim --scenario scenarios/bustabit-2019-2020-tiny.csv /home/remyers/github/bitcoin-core/test/config.ini results/ 

utxotargets.json file:

{
    "buckets": [
        {
            "start_satoshis": 10000,
            "end_satoshis": 25000,
            "target_utxo_count": 150
        },
        {
            "start_satoshis": 50000,
            "end_satoshis": 75000,
            "target_utxo_count": 50
        },
        {
            "start_satoshis": 200000,
            "end_satoshis": 250000,
            "target_utxo_count": 20
        },
        {
            "start_satoshis": 1000000,
            "end_satoshis": 1400000,
            "target_utxo_count": 5
        }
    ]
}


