# 🍔 Aaltoes CV1 submission 🍔

Using the EITLNet architecture.

Team members (teamname: `here for burgers`):

- Eerik Alamikkotervo [GitHub](https://github.com/eerik98)

- Henrik Toikka [GitHub](https://github.com/htoik)

Competition is accessible on [Kaggle](https://www.kaggle.com/competitions/aaltoes-2025-computer-vision-v-1)

## Results

TODO

## Training procedure

We adapted the EITLNet for the task. It has a dual-branch transformer structure, with RGB and image noise inputs respectively. 


![Screenshot from 2025-03-23 17-20-02](https://github.com/user-attachments/assets/da300309-a7cc-41b5-9cfc-1fa747d0fa82)


The training was done in phases with a single RTX 3070 (8GB).
1. Created train-validation split with 90% and 10% respectively
2. To learn to discriminate forged images from pristine, we loaded the forged and original as pair (if original available).4 such pairs could be fitted to the GPU (Batch size of 4*2 images). 
3. We initialized with a pre-trained mit_b2 checkpoint from the EITLNet repository
4. Hyperparameters: lr=1e-5, Adam optimizer
5. Validation score converged after 21 epochs of training -> achieved test score 96.7
6. We further trained the model without pairwise image loading for one more epoch to replicate the test scenario more accurately -> achieved test score 97.1.

Further training with/without augmentations, with different learning rates, didn't improve performance

## Installation

```bash
git clone --recursive https://github.com/eerik98/aaltoes_cv1_submission
cd aaltoes_cv1_submission

conda env create -f config/environment.yml
./scripts/setup.sh
pip install -e .
```

## Replication of submission results

The best weights can be found in `checkpoints/`. Download them, and follow the instructions in the notebook.
