from imports import *
from load_and_run import load_and_run
##todo: change the loss to be just for tumors? or emphasize tumor weight

## generated models are based off the whole train set. so it doesnt make much sense when reducing data size. (?)


### current results are for inserting a new tumor on every slice. 
### Need to run for tumors on empty slices


if __name__ == '__main__':
    from transforms import train_transforms, val_transforms, noise_elastic_train_transforms
    from transforms import RT_train_transforms, aug_RT_train_transforms
    from transforms import VAE_train_transforms, aug_VAE_train_transforms
    from transforms import VAE_GAN_train_transforms, aug_VAE_GAN_train_transforms
    
    save_path="/home/omo23/Documents/segmentation-project/saved-tests/"
    fraction_of_train_data = [100, 70, 40, 10]
    max_epochs = 50

    ##create results csv file
    if not os.path.exists(save_path+"dice-results.csv"):
        with open(save_path+'seg_results.csv','w') as fp:
            writer = csv.writer(fp, delimiter=",")
            writer.writerow(["Test name", "fraction_of_train_data", "liver", "tumor"])


    #for testing code
#     load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/test",
#                 tr_va_split = [60,20,20],
#                 percentage_of_data = 3,
#                 number_of_epochs = 2,
#                 train_transforms = VAE_train_transforms,
#                 val_transforms = val_transforms,
#                 use_vae_data=True)
        

    for run_attempt in range(3):
        for f in fraction_of_train_data:

        # all slices 
            load_and_run(save_path = save_path + str(run_attempt) + "-" + str(f) +"-standard",
                    percentage_of_data = f,
                    number_of_epochs = max_epochs,
                    train_transforms = train_transforms,
                    val_transforms = val_transforms)
            
            load_and_run(save_path = save_path + str(run_attempt) + "-" + str(f) +"-augmented"+"-standard",
                    percentage_of_data = f,
                    number_of_epochs = max_epochs,
                    train_transforms = noise_elastic_train_transforms,
                    val_transforms = val_transforms)
            

        ## VAE 
            load_and_run(save_path = save_path + str(run_attempt) + "-" + str(f) +"-VAE",
                    percentage_of_data = f,
                    number_of_epochs = max_epochs,
                    train_transforms = VAE_train_transforms,
                    val_transforms = val_transforms)
            
            load_and_run(save_path = save_path + str(run_attempt) + "-" + str(f) +"-augmented"+"-VAE",
                    percentage_of_data = f,
                    number_of_epochs = max_epochs,
                    train_transforms = aug_VAE_train_transforms,
                    val_transforms = val_transforms)



        ## VAE-GAN 
            load_and_run(save_path = save_path + str(run_attempt) + "-" + str(f) +"-VAE_GAN",
                    percentage_of_data = f,
                    number_of_epochs = max_epochs,
                    train_transforms = VAE_GAN_train_transforms,
                    val_transforms = val_transforms)

            load_and_run(save_path = save_path + str(run_attempt) + "-" + str(f) +"-augmented"+"-VAE_GAN",
                    percentage_of_data = f,
                    number_of_epochs = max_epochs,
                    train_transforms = aug_VAE_GAN_train_transforms,
                    val_transforms = val_transforms)


        ## Real tumour inserted
            load_and_run(save_path = save_path + str(run_attempt) + "-" + str(f) +"-REAL",
                    percentage_of_data = f,
                    number_of_epochs = max_epochs,
                    train_transforms = RT_train_transforms,
                    val_transforms = val_transforms)
            
            load_and_run(save_path = save_path + str(run_attempt) + "-" + str(f) +"-augmented"+"-REAL",
                    percentage_of_data = f,
                    number_of_epochs = max_epochs,
                    train_transforms = aug_RT_train_transforms,
                    val_transforms = val_transforms)




            





        # ## filtered for tumor slices data
        #     load_and_run(save_path = save_path + str(run_attempt) + str(f) ,
        #             tr_va_split = [60,20,20],
        #             percentage_of_data = f,
        #             number_of_epochs = max_epochs,
        #             train_transforms = train_transforms,
        #             val_transforms = val_transforms,
        #             use_all_slices=False)
            
        #     load_and_run(save_path = save_path + str(run_attempt) + str(f) +"-augmented",
        #             tr_va_split = [60,20,20],
        #             percentage_of_data = f,
        #             number_of_epochs = max_epochs,
        #             train_transforms = noise_elastic_train_transforms,
        #             val_transforms = val_transforms,
        #             use_all_slices=False)
                