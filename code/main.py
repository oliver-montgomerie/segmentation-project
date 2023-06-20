from imports import *
from load_and_run import load_and_run

if __name__ == '__main__':
    from transforms import train_transforms, val_transforms, noise_elastic_train_transforms, elastic_train_transforms

    # for testing code
    load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/test",
                tr_va_split = [60,20,20],
                number_of_epochs = 2,
                train_transforms = train_transforms,
                val_transforms = val_transforms)

    fraction_of_train_data = [1, 0.75, 0.5, 0.25]
    for f in fraction_of_train_data:
        load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/"+f+"-20-20-no-transform",
                tr_va_split = [60,20,20],
                fraction_of_data = f,
                number_of_epochs = 300,
                train_transforms = train_transforms,
                val_transforms = val_transforms)
        
        load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/"+f+"-20-20-elastic-noise-transform",
                tr_va_split = [60,20,20],
                fraction_of_data = f,
                number_of_epochs = 300,
                train_transforms = noise_elastic_train_transforms,
                val_transforms = val_transforms)






    # load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/60-20-20-no-transform",
    #             tr_va_split = [60,20,20],
    #             number_of_epochs = 300,
    #             train_transforms = train_transforms,
    #             val_transforms = val_transforms)
