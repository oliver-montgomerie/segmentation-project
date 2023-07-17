from imports import *
from load_and_run import load_and_run
##todo: change the loss to be just for tumors? or emphasize tumor weight

# todo: try a test with tumors but blurred in differently
# for vae & vaegan

# choose better which data is being used
# when using generated. only select slices in training set


if __name__ == '__main__':
    from transforms import train_transforms, val_transforms, noise_elastic_train_transforms, elastic_train_transforms
    save_path="/home/omo23/Documents/segmentation-project/saved-tests/"
    fraction_of_train_data = [100, 70, 40, 10]

    max_epochs = 50

    ##create results csv file
    if not os.path.exists(save_path+"dice-results.csv"):
        with open(save_path+'seg_results.csv','w') as fp:
            writer = csv.writer(fp, delimiter=",")
            writer.writerow(["Test name", "fraction_of_train_data", "liver", "tumor"])


    ##for testing code
    # load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/test",
    #             tr_va_split = [60,20,20],
    #             number_of_epochs = 2,
    #             train_transforms = noise_elastic_train_transforms,
    #             val_transforms = val_transforms,
    #             use_vae_data=True)


    for f in fraction_of_train_data:

    ## normal data
        load_and_run(save_path = save_path + str(f) +"-normal",
                tr_va_split = [60,20,20],
                percentage_of_data = f,
                number_of_epochs = max_epochs,
                train_transforms = train_transforms,
                val_transforms = val_transforms)
        
        load_and_run(save_path = save_path + str(f) +"-augmented",
                tr_va_split = [60,20,20],
                percentage_of_data = f,
                number_of_epochs = max_epochs,
                train_transforms = noise_elastic_train_transforms,
                val_transforms = val_transforms)
        

    ## normal + slices with no tumors
        load_and_run(save_path = save_path + str(f) +"-normal"+"-all_slice",
                tr_va_split = [60,20,20],
                percentage_of_data = f,
                number_of_epochs = max_epochs,
                train_transforms = train_transforms,
                val_transforms = val_transforms,
                use_all_slices=True)
        
        load_and_run(save_path = save_path + str(f) +"-augmented"+"-all_slice",
                tr_va_split = [60,20,20],
                percentage_of_data = f,
                number_of_epochs = max_epochs,
                train_transforms = noise_elastic_train_transforms,
                val_transforms = val_transforms,
                use_all_slices=True)


    ## normal + VAE 
        load_and_run(save_path = save_path + str(f) +"-normal"+"-VAE",
                tr_va_split = [60,20,20],
                percentage_of_data = f,
                number_of_epochs = max_epochs,
                train_transforms = train_transforms,
                val_transforms = val_transforms,
                use_vae_data= True)
        
        load_and_run(save_path = save_path + str(f) +"-augmented"+"-VAE",
                tr_va_split = [60,20,20],
                percentage_of_data = f,
                number_of_epochs = max_epochs,
                train_transforms = noise_elastic_train_transforms,
                val_transforms = val_transforms,
                use_vae_data=True)
        

    ## normal + VAE-GAN 
        load_and_run(save_path = save_path + str(f) +"-normal"+"-VAE_GAN",
                tr_va_split = [60,20,20],
                percentage_of_data = f,
                number_of_epochs = max_epochs,
                train_transforms = train_transforms,
                val_transforms = val_transforms,
                use_vae_gan_data = True)
        
        load_and_run(save_path = save_path + str(f) +"-augmented"+"-VAE_GAN",
                tr_va_split = [60,20,20],
                percentage_of_data = f,
                number_of_epochs = max_epochs,
                train_transforms = noise_elastic_train_transforms,
                val_transforms = val_transforms,
                use_vae_gan_data =True)


