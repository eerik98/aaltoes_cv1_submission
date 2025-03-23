# 🍔 Aaltoes CV1 submission 🍔

Using the EITLNet architecture.

Team members (teamname: `here for burgers`):

- Eerik Alamikkotervo [GitHub](https://github.com/eerik98)

- Henrik Toikka [GitHub](https://github.com/htoik)

Competition is accessible on [Kaggle](https://www.kaggle.com/competitions/aaltoes-2025-computer-vision-v-1)

## Results

TODO

## Training procedure

The training was done in phases.

1. Created train-validation split with 90% and 10% respectively
 
2. We trained the model with `lr=1e-4` for ~20 epochs

3. Few epochs were trained with `lr=1e-5`

<!-- 4. Full-send: Final model was trained with all given data  -->

## Installation

```bash
git clone --recursive https://github.com/eerik98/aaltoes_cv1_submission
cd aaltoes_cv1_submission

conda env create -f config/environment.yml
./scripts/setup.sh
pip install -e .
```

## Replication of submission results

The best weights can be found in `best_weights/`.

```bash
# TODO
```

## Training

```bash
. scripts/source.sh
python3 train.py
```
