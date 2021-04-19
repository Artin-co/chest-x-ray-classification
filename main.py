import os
import funcs 
import load_data
import tensorflow as tf
import mlflow
import subprocess
import pandas as pd
from time import time


""" GPU set up """
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto(device_count={"GPU":1, "CPU": 10})
config.gpu_options.allow_growth = True  
config.log_device_placement = True  
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)



""" creating a ssh-tunnel to server in the background """
command = 'ssh -N -L 5000:localhost:5432 artinmajdi@data7-db1.cyverse.org &'
ssh_session = subprocess.Popen('exec ' + command, stdout=subprocess.PIPE, shell=True)



""" MLflow set up """
server, artifact = funcs.mlflow_settings()
mlflow.set_tracking_uri(server)


# Creating/Setting the experiment
experiment_name = 'label_inter_dependence'

# Line below should be commented if the experiment is already created
# If kept commented during the first run of a new experiment, the set_experiment 
# will automatically create the new experiment with local artifact storage

# mlflow.create_experiment(name=experiment_name, artifact_location=artifact)
mlflow.set_experiment(experiment_name=experiment_name)


# Loading the optimization parameters aturomatically from keras
mlflow.keras.autolog()

# Starting the MLflow 
ADD_RUN_NAME = True
if ADD_RUN_NAME:
    # When we add a run_name, it will remove the run_id. this will save the run_id on top of the ui page 
    mlflow.start_run(run_name = 'without replacing parent NaN values with 1.0 in the presense of child')
    run = mlflow.active_run()
    mlflow.set_tag('run_id',run.info.run_id)
else:
    mlflow.start_run()


""" Reading Terminal Inputs """
epochs, batch_size, max_sample = funcs.reading_terminal_inputs()
# epochs, batch_size, max_sample = 3, 32, 1000


# selecting the dataset 
dataset = 'chexpert' # 'nih' chexpert
dir = '/groups/jjrodrig/projects/chest/dataset/' + dataset + '/'

# Logging the extra parameters
mlflow.log_param('dataset',dataset)
mlflow.log_param('max_sample',max_sample)


EVALUATE = True

try:
   
    # Loading the data
    (train_dataset, valid_dataset), (train_generator, valid_generator), Info= load_data.load(dir=dir, dataset=dataset, batch_size=30, mode='train_val', max_sample=max_sample)

    mlflow.log_param('train count',len(train_generator.filenames))
    mlflow.log_param('valid count',len(valid_generator.filenames))


    # Optimization
    funcs.optimize(dir, train_dataset, valid_dataset, epochs, Info)
    
    
    # Evaluation
    if EVALUATE:
        df = funcs.evaluate(dir=dir, dataset=dataset, batch_size=200)
     
        # Save the outputs as mlflow artifact
        tm = str(int(time()))
        df.to_json(dir + 'model/test_results_' + tm + '.json')
        mlflow.log_artifact(dir + 'model/test_results_' + tm + '.json')
                
finally:

    # End mlflow session
    print('\n\nEnding mlflow session')
    mlflow.end_run()

    # End the ssh session. If this failed, we can type 'pkill ssh' in the terminal 
    print('Ending ssh session')
    ssh_session.kill()

    print('Optimization Complete')





    
   
