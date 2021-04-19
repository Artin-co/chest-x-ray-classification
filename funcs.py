import argparse
import tensorflow as tf
import load_data
import funcs
from tqdm import tqdm
import numpy as np
import pandas as pd
from math import e as e_VALUE

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
    parser.add_argument("--max_sample", help="maximum number of training samples")

    args = parser.parse_args()

    epochs = int(args.epoch) if args.epoch else 3
    bsize  = int(args.bsize) if args.bsize else 30
    max_sample  = int(args.max_sample) if args.max_sample else 1000

    return epochs, bsize, max_sample


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
    'data7_db1':  'sftp://artinmajdi:temp2_data7_b@data7-db1.cyverse.org:/home/artinmajdi/mlflow_data/artifact_store'}

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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=weighted_bce_loss(Info.class_weights), metrics=[tf.keras.metrics.binary_accuracy])

    history = model.fit(train_dataset,validation_data=valid_dataset,epochs=epochs,steps_per_epoch=Info.steps_per_epoch,validation_steps=Info.validation_steps,verbose=1,use_multiprocessing=True) #  , callbacks=func_CallBacks(dir_out)

    model.save(dir_out + '/model.h5', overwrite=True, include_optimizer=False )
    
    # model = load_model('model.h5', custom_objects={'loss': asymmetric_loss(alpha)})


def evaluate(dir: str, dataset: str='chexpert', batch_size: int=1000, pathologies: list=['apple','pear']):
    
    # Loading the data
    test_generator, Info = load_data.load(dir=dir, dataset=dataset, batch_size=batch_size, mode='test')

    # Loading the model
    model = tf.keras.models.load_model(dir + 'model/model.h5')

    # Compiling the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=funcs.weighted_bce_loss(Info.class_weights), metrics=[tf.keras.metrics.binary_accuracy])

    df = calculate_newlosses_accuracies(generator=test_generator, model=model, pathologies=Info.pathologies)

    return df


def calculate_newlosses_accuracies(generator, model, pathologies):

    x_test, y_test = next(generator)

    # Looping over all test samples
    score_values = {}
    NUM_CLASSES = y_test.shape[1]

    for ix in tqdm(range(len(generator.filenames))):

        full_path, x,y = generator.filenames[ix] , x_test[ix,...] , y_test[ix,...]
        x,y = x[np.newaxis,:] , y[np.newaxis,:]

        # Estimating the loss & accuracy for instance
        eval = model.evaluate(x=x, y=y,verbose=0)

        # predicting the labels for instance
        pred = model.predict(x=x,verbose=0)

        # Measuring the loss for each class
        loss_per_class = [ tf.keras.losses.binary_crossentropy(y[...,d],pred[...,d]) for d in range(NUM_CLASSES)]

        # saving all the infos  
        score_values[full_path] = {'full_path':full_path,'loss_avg':eval[0], 'acc_avg':eval[1], 'pred':pred[0], 'pred_binary':pred[0] > 0.5, 'truth':y[0]>0.5, 'loss':np.array(loss_per_class), 'pathologies':pathologies} 


    # converting the outputs into panda dataframe
    df = pd.DataFrame.from_dict(score_values).T

    # resetting the index to integers
    df.reset_index(inplace=True)

    # # dropping the old index column
    df = df.drop(['index'],axis=1)

    return df   


class Parent_Child():
    def __init__(self, subj_info: pd.DataFrame.dtypes={}, technique: int=0):
        """ 
        
            subject_info = {'pred':[], 'loss':[], 'pathologies':['Edema','Cardiomegaly',...]}


            1. After creating a class: 
                SPC = Parent_Child(loss_dict, pred_dict, technique)

            2. Update the parent child relationship: 
            
                SPC.set_parent_child_relationship(parent_name1, child_name_list1)
                SPC.set_parent_child_relationship(parent_name2, child_name_list2)

            3. Then update the loss and probabilities

                SPC.update_loss_pred()

            4. In order to see the updated loss and probabilities use below

                loss_new_list = SPC.loss_dict_weighted  or SPC.loss_list_weighted
                pred_new_list = SPC.pred_dict_weighted  or SPC.predlist_weighted

            IMPORTANT NOTE:

                If there are more than 2 generation; it is absolutely important to enter the subjects in order of seniority 

                gen1:                grandparent (gen1)
                gen1_subjx_children: parent      (gen2)
                gen2_subjx_children: child       (gen3)

                SPC = Parent_Child(loss_dict, pred_dict, technique)

                SPC.set_parent_child_relationship(gen1_subj1, gen1_subj1_children)
                SPC.set_parent_child_relationship(gen1_subj2, gen1_subj2_children)
                                             . . .

                SPC.set_parent_child_relationship(gen2_subj1, gen2_subj1_children)
                SPC.set_parent_child_relationship(gen2_subj2, gen2_subj2_children)
                                             . . .

                SPC.update_loss_pred()
        """

        self.subj_info = subj_info        
        self.technique = technique
        self.all_parents: dict = {}

        self.loss  = subj_info.loss
        self.pred  = subj_info.pred
        self.truth = subj_info.truth

        self._convert_inputs_list_to_dict()


    def _convert_inputs_list_to_dict(self):

        self.loss_dict = {disease:self.subj_info.loss[index] for index,disease in enumerate(self.subj_info.pathologies)} 
        self.pred_dict = {disease:self.subj_info.pred[index] for index,disease in enumerate(self.subj_info.pathologies)} 

        self.loss_dict_weighted = self.loss_dict
        self.pred_dict_weighted = self.pred_dict     


    def set_parent_child_relationship(self, parent_name: str='parent_name', child_name_list: list=[]):
        self.all_parents[parent_name] = child_name_list


    def update_loss_pred(self):
        """
            techniques:
                1: coefficinet = (1 - parent_loss)
                2: coefficinet = 1 / (2 * parent_pred)
                3: coefficient = 1 / (2 * parent_pred)

                1: loss_new = loss_old * coefficient if parent_pred < 0.5 else loss_old
                2: loss_new = loss_old * coefficient if parent_pred < 0.5 else loss_old
                3. loss_new = loss_old * coefficient
        """        

        for parent_name in self.all_parents:
            self._update_loss_for_children(parent_name)

        self._convert_outputs_to_list()


    def _convert_outputs_to_list(self):
        self.loss_new = np.array([self.loss_dict_weighted[disease] for disease in self.subj_info.pathologies])
        self.pred_new = np.array([self.pred_dict_weighted[disease] for disease in self.subj_info.pathologies])


    def _update_loss_for_children(self, parent_name: str='parent_name'):

        parent_loss = self.loss_dict_weighted[parent_name]
        parent_pred = self.pred_dict_weighted[parent_name]


        coefficinet_dict = {
            1: 1 + parent_loss,
            2: (2 * parent_pred),
            3: (2 * parent_pred),
            }
        
        coefficient = coefficinet_dict[self.technique]


        for child_name in self.all_parents[parent_name]:

            new_child_loss =  self._measure_new_child_loss(coefficient, parent_name, child_name)

            self.loss_dict_weighted[child_name] = new_child_loss

            # TODO after fixed the loss sign on the server, should add a "-" sign before "new_child_loss"  and  "self.loss_dict[child_name]"" here
            self.pred_dict_weighted[child_name] = 1 - np.power(e_VALUE , -new_child_loss)
            self.pred_dict[child_name] = 1 - np.power(e_VALUE , -self.loss_dict[child_name])


    def _measure_new_child_loss(self, coefficient: float=0.0, parent_name: str='parent_name', child_name: str='child_name'):
        
        parent_loss    = self.loss_dict_weighted[parent_name]
        old_child_loss = self.loss_dict_weighted[child_name]
        
        loss_new_dict = {
            1: old_child_loss * coefficient if parent_loss < 0.5 else old_child_loss,
            2: old_child_loss * coefficient if parent_loss < 0.5 else old_child_loss,
            3: old_child_loss * coefficient,
            }

        new_child_loss = loss_new_dict[self.technique]

        return new_child_loss



class Measure_InterDependent_Loss(Parent_Child):
    def __init__(self,score: pd.DataFrame.dtypes={}, technique: int=0):

        score['loss_new'] = score['loss']
        score['pred_new'] = score['pred']

        pathologies = score.loc[0].pathologies

        self.score = score
        self.technique = technique

        for subject_ix in tqdm(self.score.index):

            Parent_Child.__init__(self, subj_info=self.score.loc[subject_ix], technique=technique)

            self.set_parent_child_relationship(parent_name='Lung Opacity'              , child_name_list=['Pneumonia', 'Atelectasis','Consolidation','Lung Lesion'])
            
            self.set_parent_child_relationship(parent_name='Enlarged Cardiomediastinum', child_name_list=['Cardiomegaly'])

            self.update_loss_pred()

            self.score.loss_new.loc[subject_ix] = self.loss_new
            self.score.pred_new.loc[subject_ix] = self.pred_new    


def applying_new_loss_measurement_method_to_all_proposed_techniques(pathologies: list=[], score: pd.DataFrame.dtypes={}):

    """ Return:  Detailed results <=>  FR  """

    L = len(pathologies)

    accuracies = np.zeros((4,L))
    FR = list(np.zeros(4))
    for technique in range(4):

        # extracting the ouput predictions
        if technique == 0: 
            FR[technique] = score
            output = score.pred
        else: 
            FR[technique] = Measure_InterDependent_Loss(score=score, technique=technique)
            output = FR[technique].score.pred_new


        # Measuring accuracy
        func = lambda x1, x2: [ (x1[j] > 0.5) == (x2[j] > 0.5) for j in range(len(x1))]

        pred_acc = score.truth.combine(output,func=func).to_list()
        pred_acc = np.array(pred_acc).mean(axis=0)

        accuracies[technique,:] = np.floor( pred_acc*1000 ) / 10


    # converting the results to pandas dataframe
    acc_df = pd.DataFrame(accuracies, columns=pathologies) 
    acc_df['technique'] = ['original','1','2','3']
    acc_df = acc_df.set_index('technique').T

    return acc_df, FR