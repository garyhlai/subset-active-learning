# subset-active-learning

Result & analysis link: https://www.notion.so/Active-Learning-Subset-Selection-Research-656df62c6bde45c2b3ed6091f2f8f85d

## Setup & Reproducibility

- Installation: `poetry install -vv`
- Adding a dependency: `poetry add <dep> -vv` (make sure to commit the .lock file)

## Contribution

- Never push to the main branch directly (let alone force push). Always check out a branch `git checkout -b <branch_name>`
  and submit PR for review.

## Evaluation Notebook: `test.ipynb`

- Use the notebook `test.ipynb` to do model evaluation and reproduce the figures/plots in the Notion page.
- Use this notebook as an anchor to find the paths of the checkpoints.

## Wandb Runs

- The notion page includes links to the wandb runs, each of which contains the git hash to the commit used for the training run.

### Subset Sampling

- Classifier checkpoint used for sampling: `/results/checkpoints/optimal_subset_classifier_19500`

### TODO

- [ ] decide on a way to share checkpoints (S3?)
