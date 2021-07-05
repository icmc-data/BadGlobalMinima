# Running the code

Make sure you logged in to wandb with `wandb login`.

Run `python3 run_train.py -h` for full documentation; to run for defaults parameters in testing mode for example, run `python3 run_train.py --testing=True`.

To load from an exisint weights file in testing mode: `!python3 run_train.py --run-weights-name="2sazg09r" --testing` - 2sazg09r is the run path of the "sparkling-eon-44" experiment for example.
