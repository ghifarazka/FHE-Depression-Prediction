# `fhe-evaluation`

This component implements Fully Homomorphic Encryption on the depression prediction model and evaluates it.

- `ckks_params_experiment.py`: Performs an experiment to decide what are the best values to be used as the CKKS CryptoContext parameters for `base.py`.
- `base.py`: The base implementation of FHE on the depresion prediction model's inference process.
- `eval.py`: Performs evaluation on each steps defined by `base.py`, and compares it to non-FHE inference.
