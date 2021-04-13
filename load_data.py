import pandas as pd
import numpy as np
import tensorflow as tf 
    
    
def nih(dir: str):
    
    """ reading the csv tables """
    
    all_data       = pd.read_csv(dir + '/files/Data_Entry_2017_v2020.csv')
    test_list      = pd.read_csv(dir + '/files/test_list.txt',names=['Image Index'])
      
    
    
    """ Writing the relative path """ 
    
    all_data['Path'] = all_data['Image Index'].map(lambda x: 'data/' + x)
    
    
    
    """ Finding the list of all studied pathologies """
    
    all_data['Finding Labels'] = all_data['Finding Labels'].replace('No Finding','normal').map(lambda x: x.split('|'))
    # pathologies = set(list(chain(*all_data['Finding Labels'])))
   
    
    """ overwriting the order of pathologeis """
    
    pathologies = ['normal', 'Pneumonia', 'Mass', 'Pneumothorax', 'Pleural_Thickening', 'Edema', 'Cardiomegaly', 'Emphysema', 'Effusion', 'Consolidation', 'Nodule', 'Infiltration', 'Atelectasis', 'Fibrosis']



    """ Creating the pathology based columns """
    
    for name in pathologies:
        all_data[name] = all_data['Finding Labels'].map(lambda x: 1 if name in x else 0)


    """ Creating the disease vectors """    
    
    all_data['disease_vector'] = all_data[pathologies].values.tolist()
    all_data['disease_vector'] = all_data['disease_vector'].map(lambda x: np.array(x))
    
    
    
    """ Removing unnecessary columns """
    
    all_data = all_data.drop(columns=['OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x',	'y]', 'Follow-up #'])


    
    """ Delecting the pathologies with at least a minimum number of samples """
    
    # MIN_CASES = 1000
    # pathologies = [name for name in pathologies if all_data[name].sum()>MIN_CASES]
    # print('Number of samples per class ({})'.format(len(pathologies)), 
    #     [(name,int(all_data[name].sum())) for name in pathologies])
       
    
    """ Resampling the dataset to make class occurrences more reasonable """

    CASE_NUMBERS = 800
    sample_weights = all_data['Finding Labels'].map(lambda x: len(x) if len(x)>0 else 0).values + 4e-2
    sample_weights /= sample_weights.sum()
    all_data = all_data.sample(CASE_NUMBERS, weights=sample_weights)

    
    """ Separating train validation test """
    
    test      = all_data[all_data['Image Index'].isin(test_list['Image Index'])]
    train_val = all_data.drop(test.index)
    
    valid     = train_val.sample(frac=0.2)
    train     = train_val.drop(valid.index)
    
    print('train size:',train.shape)
    print('valid size:',valid.shape)
    print('test size:' ,test.shape)
    
    L = len(pathologies)
    class_weights = np.ones(L)/L
    
    return train, valid, test, pathologies, class_weights


def chexpert(dir: str):
    
    """ Selecting the pathologies """
    pathologies = ["Enlarged Cardiomediastinum" , "Cardiomegaly" , "Lung Opacity" , "Lung Lesion", "Edema" , "Consolidation" , "Pneumonia" , "Atelectasis" , "Pneumothorax" , "Pleural Effusion" , "Pleural Other" , "Fracture" , "Support Devices"]
    
    """ Loading the raw table """
    train = pd.read_csv(dir + '/train.csv')
    test  = pd.read_csv(dir + '/valid.csv')

    print('before sample-pruning')
    print('train:',train.shape)
    print('test:',test.shape)
    
    """ Label Structure
        positive (exist):            1.0
        negative (doesn't exist):   -1.0
        Ucertain                     0.0
        no mention                   NaN """
    
    
    """ Extracting the pathologies of interest """
    train = cleaning_up_dataframe(train, pathologies)
    test  = cleaning_up_dataframe(test, pathologies)


    """ Selecting a few cases """
    # train = train.iloc[:1000,:]


    """ separating the uncertain samples """
    train_uncertain = train.copy()
    for name in pathologies:
        train = train.loc[train[name]!='uncertain']
        
    train_uncertain = train_uncertain.drop(train.index)
    
    
    """ Splitting train/validatiion """
    valid = train.sample(frac=0.2)
    train = train.drop(valid.index)
    
    
    print('after sample-pruning')
    print('train (certain):',train.shape)
    print('train (uncertain):',train_uncertain.shape)
    print('valid:',valid.shape)
    print('test:',test.shape)
    
    """ Class weights """
    L = len(pathologies)
    class_weights = np.ones(L)/L
    
    return (train, train_uncertain), valid, test, pathologies, class_weights


def cleaning_up_dataframe(data, classes):
    """ Label Structure
        positive (exist):            1.0
        negative (doesn't exist):   -1.0
        Ucertain                     0.0
        no mention                   NaN """

    # changing all no mention labels to negative
    data = data[data['AP/PA']=='AP']
    data = data[data['Frontal/Lateral']=='Frontal']
    data = data.replace(np.nan,-1.0)

    # appending the path to each sample
    data = data[ ['Path'] + classes ]

    for column in classes:
        # data[column] = data[column].astype(int)
        data[column] = data[column].replace(1,'pos')
        data[column] = data[column].replace(-1,'neg')
        data[column] = data[column].replace(0,'uncertain')
        
    return data

def removing_uncertain_samples(data, pathologies):
    """ Label Structure
        positive (exist):            1.0
        negative (doesn't exist):   -1.0
        Ucertain                     0.0
        no mention                   NaN """
                
    for name in pathologies:
        data = data.loc[data[name]!='uncertain']

    # changing negative from -1.0 to 0.0
    # data = data.replace(-1.0,0.0)

    return data


def load(dir='', dataset='nih', batch_size=30, mode='train_val'): # mode='train_val' , 'test

    class Info_Class:
        def __init__(self, pathologies, class_weights, target_size, steps_per_epoch, validation_steps):
            self.pathologies      = pathologies
            self.class_weights    = class_weights
            self.target_size      = target_size
            self.steps_per_epoch  = steps_per_epoch
            self.validation_steps = validation_steps
            
            
    if dataset == 'nih':
        train, valid, test, pathologies, class_weights = nih(dir)
    elif dataset == 'chexpert':
        (train, _), valid, test, pathologies, class_weights = chexpert(dir)
        


    """ Keras Generator """
    generator = tf.keras.preprocessing.image.ImageDataGenerator()

    target_size =  (224,224) # (64,64)  #
    class_mode='raw'
    color_mode = 'rgb'
    y_col = list(pathologies) #'disease_vector'


    if mode == 'train_val':
        
        train_generator = generator.flow_from_dataframe(dataframe=train, x_col='Path', y_col=y_col,color_mode=color_mode,directory=dir, target_size=target_size, batch_size=10000, class_mode=class_mode, shuffle=False)

        valid_generator = generator.flow_from_dataframe(dataframe=valid, x_col='Path', y_col=y_col,color_mode=color_mode,directory=dir, target_size=target_size, batch_size=10000, class_mode=class_mode, shuffle=False)  
        
        (x_train, y_train) = next(train_generator)
        steps_per_epoch = int(x_train.shape[0]/batch_size)
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.repeat().batch(batch_size)

        
        (x_valid, y_valid) = next(valid_generator)
        validation_steps = int(x_valid.shape[0]/batch_size)
        valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)) 
        valid_data = valid_data.repeat().batch(batch_size)

        Info = Info_Class(pathologies, class_weights, target_size, steps_per_epoch, validation_steps)

        return train_data, valid_data, Info



    elif mode == 'test':
        test_generator = generator.flow_from_dataframe(dataframe=test, x_col='Path', y_col=y_col,color_mode=color_mode,directory=dir, target_size=target_size, batch_size=1, class_mode=class_mode, shuffle=False)

        Info = Info_Class(pathologies, class_weights, target_size, '', '')

        return test_generator, Info

