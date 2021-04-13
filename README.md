# Installation

## installing the package requirements
    conda env create -f requirements.yaml -n env_name

    or 

    docker pull artinmajdi/miniconda-cuda-tensorflow:latest
    
    or

    conda install -c anaconda keras tensorflow-gpu
    conda install -c anaconda numpy pandas matplotlib 
    conda install -c anaconda scikit-learn scikit-image
    conda install -c anaconda psycopg2 git
    pip install mlflow==1.12.1
    pip install pysftp

