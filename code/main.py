from imports import *
from load_and_run import load_and_run
##todo: change the loss to be just for tumors? or emphasize tumor weight

if __name__ == '__main__':
    from transforms import train_transforms, val_transforms, noise_elastic_train_transforms, elastic_train_transforms

    ##for testing code
    # load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/test",
    #             tr_va_split = [60,20,20],
    #             number_of_epochs = 2,
    #             train_transforms = noise_elastic_train_transforms,
    #             val_transforms = val_transforms,
    #             use_vae_data=True)


    ## normal data
    fraction_of_train_data = [1.0, 0.70, 0.40, 0.10]
    for f in fraction_of_train_data:
        load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/"+str(int(f*60))+"-20-20-no-transform",
                tr_va_split = [60,20,20],
                fraction_of_data = f,
                number_of_epochs = 50,
                train_transforms = train_transforms,
                val_transforms = val_transforms)
        
        load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/"+str(int(f*60))+"-20-20-elastic-noise-transform",
                tr_va_split = [60,20,20],
                fraction_of_data = f,
                number_of_epochs = 50,
                train_transforms = noise_elastic_train_transforms,
                val_transforms = val_transforms)
        

    ## normal + slices with no tumors
    fraction_of_train_data = [1.0]
    for f in fraction_of_train_data:
        load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/empty-"+str(int(f*60))+"-20-20-no-transform",
                tr_va_split = [60,20,20],
                fraction_of_data = f,
                number_of_epochs = 50,
                train_transforms = train_transforms,
                val_transforms = val_transforms,
                use_empty_slices=True)
        
        load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/empty-"+str(int(f*60))+"-20-20-elastic-noise-transform",
                tr_va_split = [60,20,20],
                fraction_of_data = f,
                number_of_epochs = 50,
                train_transforms = noise_elastic_train_transforms,
                val_transforms = val_transforms,
                use_empty_slices=True)


    ## normal + VAE 
    fraction_of_train_data = [1.0, 0.70, 0.40, 0.10]
    for f in fraction_of_train_data:
        load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/VAE-"+str(int(f*60))+"-20-20-no-transform",
                tr_va_split = [60,20,20],
                fraction_of_data = f,
                number_of_epochs = 50,
                train_transforms = train_transforms,
                val_transforms = val_transforms,
                use_vae_data= True)
        
        load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/VAE-"+str(int(f*60))+"-20-20-elastic-noise-transform",
                tr_va_split = [60,20,20],
                fraction_of_data = f,
                number_of_epochs = 50,
                train_transforms = noise_elastic_train_transforms,
                val_transforms = val_transforms,
                use_vae_data=True)
        

    ## normal + VAE-GAN 
    fraction_of_train_data = [1.0, 0.70, 0.40, 0.10]
    for f in fraction_of_train_data:
        load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/VAE-GAN-"+str(int(f*60))+"-20-20-no-transform",
                tr_va_split = [60,20,20],
                fraction_of_data = f,
                number_of_epochs = 50,
                train_transforms = train_transforms,
                val_transforms = val_transforms,
                use_vae_gan_data = True)
        
        load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/VAE-GAN-"+str(int(f*60))+"-20-20-elastic-noise-transform",
                tr_va_split = [60,20,20],
                fraction_of_data = f,
                number_of_epochs = 50,
                train_transforms = noise_elastic_train_transforms,
                val_transforms = val_transforms,
                use_vae_gan_data =True)


