import keras
# from keras.applications.densenet import DenseNet121
import argparse
import tensorflow as tf

def func_CallBacks(Dir_Save):
    mode = 'min'
    monitor = 'val_loss'

    # checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath= Dir_Save + '/best_model_weights.h5', monitor=monitor , verbose=1, save_best_only=True, mode=mode)
    # Reduce_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1, min_delta=0.005 , patience=10, verbose=1, save_best_only=True, mode=mode , min_lr=0.9e-5 , )
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=4, verbose=1, mode=mode, baseline=0, restore_best_weights=True)
    # CSVLogger = tf.keras.callbacks.CSVLogger(Dir_Save + '/results.csv', separator=',', append=False)

    return [EarlyStopping] # [checkpointer  , EarlyStopping , CSVLogger]


def reading_terminal_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", help="number of epochs")
    parser.add_argument("--bsize", help="batch size")

    args = parser.parse_args()

    epochs = int(args.epoch) if args.epoch else 2
    bsize  = int(args.bsize) if args.bsize else 30

    return epochs, bsize


def mlflow_settings():
    """
    RUN UI with postgres and HPC:
    REMOTE postgres server:
        # connecting to remote server through ssh tunneling
        ssh -L 5000:localhost:5432 artinmajdi@data7-db1.cyverse.org

        # using the mapped port and localhost to view the data
        mlflow ui --backend-store-uri postgresql://artinmajdi:1234@localhost:5000/chest_db --port 6789

    RUN directly from GitHub or show experiments/runs list:

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    
    mlflow runs list --experiment-id <id>

    mlflow run                 --no-conda --experiment-id 5 -P epoch=2 https://github.com/artinmajdi/mlflow_workflow.git -v main
    mlflow run mlflow_workflow --no-conda --experiment-id 5 -P epoch=2
    
    PostgreSQL server style
        server = f'{dialect_driver}://{username}:{password}@{ip}/{database_name}' """

    postgres_connection_type = { 'direct':     ('5432', 'data7-db1.cyverse.org'),
                                    'ssh-tunnel': ('5000', 'localhost')
                                }

    port, host = postgres_connection_type['ssh-tunnel'] # 'direct' , 'ssh-tunnel'
    username = "artinmajdi"
    password = '1234'
    database_name = "chest_db" # chest_db
    dialect_driver = 'postgresql'

    server = f'{dialect_driver}://{username}:{password}@{host}:{port}/{database_name}'

    Artifacts = {
    'hpc':        'sftp://mohammadsmajdi@filexfer.hpc.arizona.edu:/home/u29/mohammadsmajdi/projects/mlflow/artifact_store',
    'data7_db1':  'sftp://artinmajdi:temp_data7@data7-db1.cyverse.org:/home/artinmajdi/mlflow_data/artifact_store'}

    artifact = Artifacts['data7_db1']
    
    return server, artifact


def architecture(name, input_shape,num_classes):
    
    if  name == 'densenet':
        model = tf.keras.applications.densenet.DenseNet121(weights='imagenet',include_top=False,input_tensor=tf.keras.layers.Input(input_shape),input_shape=input_shape, pooling='avg') # ,classes=num_classes

        # KK = tf.keras.layers.GlobalAveragePooling2D()
        KK = tf.keras.layers.Dense(num_classes, activation='sigmoid', name='predictions')(model.output)
        return tf.keras.models.Model(inputs=model.input,outputs=KK)    
        
    else:

        inputs = tf.keras.layers.Input(input_shape)

        model = tf.keras.layers.Conv2D(4, kernel_size=(3,3), activation='relu')(inputs)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.MaxPooling2D(2,2)(model)
        
        model = tf.keras.layers.Conv2D(8, kernel_size=(3,3), activation='relu')(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.MaxPooling2D(2,2)(model)

        model = tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu')(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.MaxPooling2D(2,2)(model)

        model = tf.keras.layers.Flatten()(model)
        model = tf.keras.layers.Dense(32, activation='relu')(model)
        model = tf.keras.layers.Dense(num_classes , activation='softmax')(model)

        return tf.keras.models.Model(inputs=[inputs], outputs=[model])


# def func_average(loss, NUM_CLASSES):
#     return tf.divide(tf.reduce_sum(loss), tf.cast(NUM_CLASSES,tf.float32))

def weighted_bce_loss(W):
    
    def func_loss(y_true,y_pred):
        NUM_CLASSES = y_pred.shape[1]
        loss = [ W[d]*tf.keras.losses.binary_crossentropy(y_true[...,d],y_pred[...,d]) for d in range(NUM_CLASSES)]
        return tf.divide(tf.reduce_sum(loss), tf.cast(NUM_CLASSES,tf.float32))

    return func_loss


def optimize(dir, train_dataset, valid_dataset, epochs, Info):
    
    """ architecture  """
    num_classes = len(Info.pathologies)
    input_shape = list(Info.target_size) + [3] 
    dir_out     = dir + '/model'

    model = architecture('densenet', input_shape, num_classes)
    
    # model.summary()


    """ optimization """
    # weighted_bce_loss(class_weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=weighted_bce_loss(Info.class_weights), metrics=[tf.keras.metrics.binary_accuracy])

    history = model.fit(train_dataset,validation_data=valid_dataset,epochs=epochs,steps_per_epoch=Info.steps_per_epoch,validation_steps=Info.validation_steps,verbose=1,use_multiprocessing=True) #  , callbacks=func_CallBacks(dir_out)

    model.save(dir_out + '/model2.h5', overwrite=True, include_optimizer=False )
    
    # model = load_model('model.h5', custom_objects={'loss': asymmetric_loss(alpha)})



def update_loss(measured_loss: dict={}, predicted_probability: dict={}, child: str='Atelectasis', parent: str ='Pneumonia', method=1):
    
    if method <= 3:
        if method == 1:
            weight = 1 + measured_loss[parent] if predicted_probability[parent] < 0.5 else 1
        
        elif method == 2:
            weight = 1 / (2 * predicted_probability[parent]) if predicted_probability[parent] < 0.5 else 1

        elif method == 3:
            weight = 1 / (2 * predicted_probability[parent])
 
        new_loss_child = measured_loss[child] * weight

    elif method == 'fitting':
        # fitting a function for: p_child    = g(q_child, q_parent)
        # then returning          qhat_child = g(q_child, q_parent)
        # new_loss_child = - ( p_child * log (qhat_child) + (1-p_child) * log( 1-qhat_child ) )
        new_loss_child = measured_loss[child]


    return new_loss_child


def measure_uncertainty():
    return 1