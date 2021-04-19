# How to run
entry_points:
    main:
        parameters:
            epoch: {type: int, default: 3}
            bsize: {type: int, default: 30}
            max_sample: {type: int, default: 1000}
        command: python main.py --epoch {epoch} --bsize {bsize} --max_sample {max_sample}
        

# Installation

### Using conda 
    conda env create -f requirements.yaml -n env_name

### Using docker

    docker pull artinmajdi/miniconda-cuda-tensorflow:latest
    
### Installing  packages manually

    conda install -c anaconda keras tensorflow-gpu
    conda install -c anaconda numpy pandas matplotlib 
    conda install -c anaconda scikit-learn scikit-image
    conda install -c anaconda psycopg2 git
    pip install mlflow==1.12.1
    pip install pysftp
