from logging import error, warning
import pandas as pd
import numpy as np
import tensorflow as tf 
    
    
def nih(dir: str, max_sample: int):
    
    """ reading the csv tables """    
    all_data       = pd.read_csv(dir + '/files/Data_Entry_2017_v2020.csv')
    test_list      = pd.read_csv(dir + '/files/test_list.txt',names=['Image Index'])
      
    
    
    """ Writing the relative path """     
    all_data['Path']      = 'data/' + all_data['Image Index']
    all_data['full_path'] = dir +'/data/' + all_data['Image Index']
    
    
    
    """ Finding the list of all studied pathologies """    
    all_data['Finding Labels'] = all_data['Finding Labels'].map(lambda x: x.split('|'))
    # pathologies = set(list(chain(*all_data['Finding Labels'])))
   

    
    """ overwriting the order of pathologeis """    
    pathologies = ['No Finding', 'Pneumonia', 'Mass', 'Pneumothorax', 'Pleural_Thickening', 'Edema', 'Cardiomegaly', 'Emphysema', 'Effusion', 'Consolidation', 'Nodule', 'Infiltration', 'Atelectasis', 'Fibrosis']



    """ Creating the pathology based columns """    
    for name in pathologies:
        all_data[name] = all_data['Finding Labels'].map(lambda x: 1 if name in x else 0)

        

    """ Creating the disease vectors """        
    all_data['disease_vector'] = all_data[pathologies].values.tolist()
    all_data['disease_vector'] = all_data['disease_vector'].map(lambda x: np.array(x))  
   
    
   
    """ Selecting a few cases """    
    all_data = all_data.iloc[:max_sample,:]
    
    
    
    """ Removing unnecessary columns """    
    # all_data = all_data.drop(columns=['OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x',	'y]', 'Follow-up #'])


    
    """ Delecting the pathologies with at least a minimum number of samples """    
    # MIN_CASES = 1000
    # pathologies = [name for name in pathologies if all_data[name].sum()>MIN_CASES]
    # print('Number of samples per class ({})'.format(len(pathologies)), 
    #     [(name,int(all_data[name].sum())) for name in pathologies])
       
        
    
    """ Resampling the dataset to make class occurrences more reasonable """
    # CASE_NUMBERS = 800
    # sample_weights = all_data['Finding Labels'].map(lambda x: len(x) if len(x)>0 else 0).values + 4e-2
    # sample_weights /= sample_weights.sum()
    # all_data = all_data.sample(CASE_NUMBERS, weights=sample_weights)

    
    
    """ Separating train validation test """    
    test      = all_data[all_data['Image Index'].isin(test_list['Image Index'])]
    train_val = all_data.drop(test.index)

    valid     = train_val.sample(frac=0.2,random_state=1)
    train     = train_val.drop(valid.index)

    print('after sample-pruning')
    print('train size:',train.shape)
    print('valid size:',valid.shape)
    print('test size:' ,test.shape) 
    
    
    
    """ Class weights """
    L = len(pathologies)
    class_weights = np.ones(L)/L
    
    
    
    return train, valid, test, pathologies, class_weights


def chexpert(dir: str, max_sample: int):
    
    """ Selecting the pathologies """    
    pathologies = ["No Finding", "Enlarged Cardiomediastinum" , "Cardiomegaly" , "Lung Opacity" , "Lung Lesion", "Edema" , "Consolidation" , "Pneumonia" , "Atelectasis" , "Pneumothorax" , "Pleural Effusion" , "Pleural Other" , "Fracture" , "Support Devices"]
    
    
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
    
    """ Adding full directory """    
    train['full_path'] = dir +'/' + train['Path']
    test['full_path'] = dir +'/' + test['Path']
    
    
    
    """ Extracting the pathologies of interest """    
    train = cleaning_up_dataframe(train, pathologies, 'train')
    test  = cleaning_up_dataframe(test, pathologies , 'test')


    """ Selecting a few cases """    
    train = train.iloc[:max_sample,:]


    """ Separating the uncertain samples """    
    train_uncertain = train.copy()
    for name in pathologies:
        train = train.loc[train[name]!='uncertain']
        
    train_uncertain = train_uncertain.drop(train.index)

        
    """ Splitting train/validatiion """    
    valid = train.sample(frac=0.2,random_state=1)
    train = train.drop(valid.index)
    
    
    print('\nafter sample-pruning')
    print('train (certain):',train.shape)
    print('train (uncertain):',train_uncertain.shape)
    print('valid:',valid.shape)
    print('test:',test.shape,'\n')
    
    
    
    """ Changing classes from string to integer """    
    train = train.replace('pos',1).replace('neg',0)
    valid = valid.replace('pos',1).replace('neg',0)
    test  = test.replace('pos',1).replace('neg',0)

    
    """ Changing the NaN values for parents with at lease 1 TRUE child to TRUE """
    # train = replacing_parent_nan_values_with_one_if_child_exist(train)
    # valid = replacing_parent_nan_values_with_one_if_child_exist(valid)

    
    """ Class weights """    
    L = len(pathologies)
    class_weights = np.ones(L)/L
    
    return (train, train_uncertain), valid, test, pathologies, class_weights


def cleaning_up_dataframe(data, pathologies, mode):
    """ Label Structure
        positive (exist):            1.0
        negative (doesn't exist):   -1.0
        Ucertain                     0.0
        no mention                   NaN """

    # changing all no mention labels to negative
    data = data[data['AP/PA']=='AP']
    data = data[data['Frontal/Lateral']=='Frontal']

    # according to CheXpert paper, we can assume all pathologise are negative when no finding label is True
    # TODO should I remove these cases instead of this?
    no_finding_indexes = data[data['No Finding']==1].index
    for disease in pathologies:
        data.loc[no_finding_indexes, disease] = -1.0


    # Treat all other NaN s as negative
    # TODO commoented temporarily
    data = data.replace(np.nan,-1.0)


    # renaming the pathologeis to 'neg' 'pos' 'uncertain'
    for column in pathologies:
        
        data[column] = data[column].replace(1,'pos')
        
        if mode == 'train':
            data[column] = data[column].replace(-1,'neg')
            data[column] = data[column].replace(0,'uncertain')
        elif mode == 'test':
            data[column] = data[column].replace(0,'neg')
            
        
    return data



def replacing_parent_nan_values_with_one_if_child_exist(data: pd.DataFrame):

    """     parent ->
                - child
 
            Lung Opacity -> 

                - Pneuomnia
                - Atelectasis
                - Edema
                - Consolidation
                - Lung Lesion

            Enlarged Cardiomediastinum -> 

                - Cardiomegaly       """


    func = lambda x1, x2: 1.0 if np.isnan(x1) and x2==1.0 else x1

    for child_name in ['Pneumonia','Atelectasis','Edema','Consolidation','Lung Lesion']:

        data['Lung Opacity'] = data['Lung Opacity'].combine(data[child_name], func=func)


    for child_name in ['Cardiomegaly']:

        data['Enlarged Cardiomediastinum'] = data['Enlarged Cardiomediastinum'].combine(data[child_name], func=func)

    return data



def load(dir='', dataset='chexpert', batch_size=30, mode='train_val', max_sample=100000): # mode='train_val' , 'test

    class Info_Class:
        def __init__(self, pathologies, class_weights, target_size, steps_per_epoch, validation_steps):
            self.pathologies      = pathologies
            self.class_weights    = class_weights
            self.target_size      = target_size
            self.steps_per_epoch  = steps_per_epoch
            self.validation_steps = validation_steps
            
            
    if dataset == 'nih':
        train, valid, test, pathologies, class_weights = nih(dir,max_sample)
    elif dataset == 'chexpert':
        (train, _), valid, test, pathologies, class_weights = chexpert(dir,max_sample)
        


    """ Keras Generator """
    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    target_size =  (224,224) # (64,64)
    class_mode='raw'
    color_mode = 'rgb' # this is actually grayscale. but because densenet input is rgb, is set to rgb
    y_col = list(pathologies)
    output_shapes = ([None,224,224,3],[None,len(y_col)]) # [None,len(y_col)] # ( [None] + list(target_size) + [3],[None,len(y_col)] )
    output_types= (tf.float32,tf.float32)
    

    # thi
    if mode == 'train_val':
        
        train_generator = generator.flow_from_dataframe(dataframe=train, x_col='Path', y_col=y_col,color_mode=color_mode,directory=dir, target_size=target_size, batch_size=batch_size, class_mode=class_mode, shuffle=False)

        valid_generator = generator.flow_from_dataframe(dataframe=valid, x_col='Path', y_col=y_col,color_mode=color_mode,directory=dir, target_size=target_size, batch_size=batch_size, class_mode=class_mode, shuffle=False)  

        # (x_train, y_train) = next(train_generator)
        steps_per_epoch = int(len(train_generator.filenames)/batch_size)
        # train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        # train_data = train_data.repeat().batch(batch_size)

        train_data = tf.data.Dataset.from_generator(lambda: train_generator,output_types=output_types)

        # (x_valid, y_valid) = next(valid_generator)
        validation_steps = int(len(train_generator.filenames)/batch_size)
        # valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)) 
        # valid_data = valid_data.repeat().batch(batch_size)

        valid_data = tf.data.Dataset.from_generator(lambda: valid_generator,output_types=output_types)
        
        Info = Info_Class(pathologies, class_weights, target_size, steps_per_epoch, validation_steps)

        return (train_data, valid_data), (train_generator, valid_generator), Info
        # return (train, valid), (train_generator, valid_generator), Info



    elif mode == 'test':
        test_generator = generator.flow_from_dataframe(dataframe=test, x_col='Path', y_col=y_col,color_mode=color_mode,directory=dir, target_size=target_size, batch_size=batch_size, class_mode=class_mode, shuffle=False)

        Info = Info_Class(pathologies, class_weights, target_size, '', '')

        return test_generator, Info

