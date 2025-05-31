# psc-iv-curves

WIP for UofT-NUS research project

## Dataset

Refer to `explanation_of_data.pdf` for details on the dataset. Make sure you have `git-lfs` installed (`brew install git-lfs` on Unix) and run `git lfs pull` to download the dataset.

Differences in inference time (single physics sample, running on the EOS setup):

- Mamba: 0.099s
- Transformer: 0.080s
- RNN based: 0.052

All tests are performed on the T4 GPU.
