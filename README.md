SSP Metamodel - Surrogate ML model + Optimization routine

## Getting Started
To get started, follow these steps:

1. **Clone the repository** (if you haven't already):

    ```bash
    git clone https://github.com/yourusername/ssp_metamodel.git
    cd ssp_metamodel
    ```

2. **Create a new Conda environment**:

    ```bash
    conda create -n ssp_metamodel_env python=3.11
    conda activate ssp_metamodel_env
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

You're now ready to use the SSP Metamodel!

## Important Files

- [notebooks/etl.ipynb](notebooks/etl.ipynb): Notebook where the `lhs_samples` data and the SISEPUEDE emission output data are merged. The training dataframe is created here.
- [notebooks/model_draft_gb_2.ipynb](notebooks/model_draft_gb_2.ipynb): Notebook with the machine learning pipeline to train a gradient boosting model to predict emissions per subsector.
