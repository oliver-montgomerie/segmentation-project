from imports import *
from load_and_run import load_and_run

if __name__ == '__main__':
    from transforms import train_transforms, val_transforms

    # for testing code
    load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/test2",
                tr_va_split = [60,20,20],
                number_of_epochs = 200,
                train_transforms = train_transforms,
                val_transforms = val_transforms)

    
    # load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/45-20-20-no-transform",
    #             tr_va_split = [60,20,20],
    #             fraction_of_data = 0.75,
    #             number_of_epochs = 300,
    #             train_transforms = train_transforms,
    #             val_transforms = val_transforms)

    
    # load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/15-20-20-no-transform",
    #             tr_va_split = [60,20,20],
    #             fraction_of_data = 0.25,
    #             number_of_epochs = 300,
    #             train_transforms = train_transforms,
    #             val_transforms = val_transforms)
    
    # from transforms import noise_train_transforms
    # load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/60-20-20-noise-transform",
    #             tr_va_split = [60,20,20],
    #             number_of_epochs = 300,
    #             train_transforms = noise_train_transforms,
    #             val_transforms = noise_train_transforms)
    
    
    # from transforms import noise_elastic_train_transforms
    # load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/60-20-20-elastic-noise-transform",
    #             tr_va_split = [60,20,20],
    #             number_of_epochs = 300,
    #             train_transforms = noise_elastic_train_transforms,
    #             val_transforms = noise_elastic_train_transforms)

    ### Ran already
    # from transforms import train_transforms, val_transforms
    # load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/60-20-20-no-transform",
    #             tr_va_split = [60,20,20],
    #             number_of_epochs = 300,
    #             train_transforms = train_transforms,
    #             val_transforms = val_transforms)

    ### ran already
    # load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/30-20-20-no-transform",
    #             tr_va_split = [60,20,20],
    #             fraction_of_data = 0.50,
    #             number_of_epochs = 300,
    #             train_transforms = train_transforms,
    #             val_transforms = val_transforms)

    ### ran already
    # from transforms import elastic_train_transforms
    # load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/60-20-20-elastic-transform",
    #             tr_va_split = [60,20,20],
    #             number_of_epochs = 300,
    #             train_transforms = elastic_train_transforms,
    #             val_transforms = elastic_train_transforms)