# psc-iv-curves

WIP for UofT-NUS research project

Description to be written

## Dataset

Refer to `explanation_of_data.pdf` for details on the dataset. Make sure you have `git-lfs` installed (`brew install git-lfs` on Unix) and run `git lfs pull` to download the dataset.

## Running the Code

The train and inference scripts are located in the `src` directory. Run them as follows:

```bash
python -m src.script.your_script_name
```

Trying to run the scripts directly (e.g., `python src/script/your_script_name.py`) will not work due to the way the modules are structured (might be changed in the future).

## Preprocessing

The preprocessing script is located in `src/utils/preprocess.py`. It processes the dataset and saves the preprocessed data in the `dataset` directory. I will write more about this later or just refer to the paper when we publish.

## Models

Models are defined in the `src/model` directory. The following models are available:

- `transformer.py`: Transformer model
- `rnn.py`: RNN-based model (BiLSTM)
- `mlp.py`: MLP model (two heads and one head versions)
- `cvae.py`: Conditional Variational Autoencoder (CVAE) model

Old models not actively developed:

- `mamba.py`: Mamba model

## Random Notes

Differences in inference time (single physics sample, running on the EOS setup):

- Mamba: 0.099s
- Transformer: 0.080s
- RNN based: 0.052

All tests are performed on the T4 GPU.
