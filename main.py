import os
import funcs 
import load_data
import tensorflow as tf
import mlflow
import subprocess


""" GPU set up """
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.compat.v1.ConfigProto(device_count={"GPU":1, "CPU": 10})
config.gpu_options.allow_growth = True  
config.log_device_placement = True  
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)



""" creating a ssh-tunnel to server in the background """
command = 'ssh -N -L 5000:localhost:5432 <username>@<remote-server-address> &'
ssh_session = subprocess.Popen('exec ' + command, stdout=subprocess.PIPE, shell=True)



""" MLflow set up """
server, artifact = funcs.mlflow_settings()
mlflow.set_tracking_uri(server)


# Creating/Setting the experiment
experiment_name = '/chexpert_d1'

# Line below should be commented if the experiment is already created
# If kept commented during the first run of a new experiment, the set_experiment 
# will automatically create the new experiment with local artifact storage

mlflow.create_experiment(name=experiment_name, artifact_location=artifact)
mlflow.set_experiment(experiment_name=experiment_name)


# Loading the optimization parameters aturomatically from keras
mlflow.keras.autolog()

# Starting the MLflow 
mlflow.start_run()



""" Reading Terminal Inputs """
epochs, batch_size = funcs.reading_terminal_inputs()
# epochs, batch_size = 3, 32


# selecting the dataset 
dataset = 'chexpert' # 'nih'

try:
   
    dir = '/groups/jjrodrig/projects/chest/dataset/' + dataset + '/'
    train_dataset, valid_dataset, Info = load_data.load(dir=dir, dataset=dataset, batch_size=30, mode='train_val')

    funcs.optimize(dir, train_dataset, valid_dataset, epochs, Info)
    
    mlflow.end_run()
    
finally:

    # ending the ssh session. If this failed, we can type 'pkill ssh' in the terminal to kill all ssh sessions
    ssh_session.kill()
    print('Finished')





    
   