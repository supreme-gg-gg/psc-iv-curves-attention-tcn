# PSC IV Curves Reconstruction with Attention TCN

This repo contains experiment code for the paper "Reconstruction of IV Characteristics of Perovskite Solar Cells using Physics-Informed Temporal Convolution Attention Networks" (will be published soon).

This is not the official code for the paper, which is available [in this repo instead](https://github.com/memo-ozdincer/PINN-iV-curve-reconstruction) by my partner. The official repo contains up-to-date code, results, and a more detailed writeup.

> This repo is not actively maintained! You should checkout the official code repo if you wish to use our code in your projects / research.

## Dataset

Refer to `explanation_of_data.pdf` for details on the dataset. Make sure you have `git-lfs` installed (`brew install git-lfs` on Unix) and run `git lfs pull` to download the dataset.

## Structure

```plaintext
psc-iv-curves-attention-tcn/
├── notebooks/      # Visualisation and analysis
├── tcn-attention/  # Latest TCAN model
├── src/            # Experimental models and utils
├── script/         # Training and inference scripts (to be updated)
├── checkpoints/    # Will upload later
└── results/        # Will upload later
```

## Running the Code

The train and inference scripts are located in the `script` directory. Run them as follows:

```bash
python -m script.your_script_name
```

Trying to run the scripts directly (e.g., `python script/your_script_name.py`) will not work due to the way the modules are structured (might be changed in the future).

## Preprocessing

The preprocessing script is located in `src/utils/preprocess.py` or `tcn-attention/utils/preprocessor.py`. It processes the dataset and saves the preprocessed data in the `dataset` directory. I will write more about this later or just refer to the paper when we publish.

### Handling variable sequence lengths

We use either interpolation (linear / pchip) or padding to handle variable sequence lengths. Multiple methods are available depending on the problem formulation and model requirements.

### Positional encoding

Regular transformer uses sinusoidal positional encoding but TCN family of models use Fourier feature positional encoding (either logspaced or Gaussian) as additional feature embedding.

## Models Architecture

Models are defined in the `src/model` directory:

- `transformer.py`: Transformer decoder model (a few variants with different EOS and heads)
- `rnn.py`: RNN-based model (BiLSTM)
- `mlp.py`: MLP model (two heads and one head versions)
- `cvae.py`: Conditional Variational Autoencoder (CVAE) model
- Ones not actively maintained such as Mamba, autoencoder with MLP, embedding models

Latest approach with TCN-based models are in `tcn-attention/model` directory:

- `basic_tcn.py`: Temporal Convolutional Network (TCN) with dilated, causal convolutions
- `temporal_attention_tcn.py`: Masked multihead self attention between TCN layers
- `neighbor_attention_tcn.py`: Using neighbour attention (NA) to reduce model size

## Physics-informed loss functions

See `tcn-attention/utils/trainer.py` and `src/utils/loss_functions.py` for details.

## Results

For now refer to the offical code repo for a detailed explanation.

## Citation

You should cite the official code repo. This research work is done by Jet Chiang and Memo Ozdincer from University of Toronto, under supervision by Prof. Erik Birgersson from National University of Singapore.
