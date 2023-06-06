from pickle import FALSE
from imports import *

#todo: make txt save data in the folder
# add batch size and epochs along with the #train /#test

def load_and_run(save_path = "", tr_va_split=[80,20], number_of_epochs = 1000):

    num_workers = 16
    #Data loading
    data_dir = "/data/datasets/Liver/LiTS2017"   
    train_images = sorted(glob.glob(os.path.join(data_dir, "Volumes", "*.nii")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "Segmentations", "*.nii")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    data_dir = "/data/datasets/Liver/LiTS2017/Test"   
    test_images = sorted(glob.glob(os.path.join(data_dir, "Volumes", "*.nii")))
    test_labels = sorted(glob.glob(os.path.join(data_dir, "Segmentations", "*.nii")))
    test_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]

    #no_files = len(data_dicts) #todo: test this line
    no_files = 131
    number_of_training = (tr_va_split[0] * no_files) // 100
    number_of_validation = (tr_va_split[1] * no_files) // 100
    number_of_test = no_files - number_of_training - number_of_validation

    train_files = data_dicts[0:number_of_training]
    val_files = data_dicts[number_of_training:number_of_validation]

    print("Number of train files:", len(train_files), "Number of val files:", len(val_files), "Number of test files:", len(test_files))

    #Load data transforms
    from transforms import train_transforms, val_transforms, test_transforms

    ###datasets
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=num_workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)
    test_ds = Dataset(data=test_files, transform=test_transforms)

    ###dataloaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=num_workers)

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.9)     #ReduceLROnPlateau
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    print("Created model, loss, optim ,dice")

    #Training loop
    from training_loop import training_loop
    epoch_loss_values, metric_values = training_loop(
                    model = model,
                    train_loader = train_loader,
                    val_loader = val_loader,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    loss_function = loss_function,
                    dice_metric = dice_metric,
                    device = device,
                    max_epochs = number_of_epochs,
                    model_path_and_name = os.path.join(save_path, "best_metric_model.pth"),
                    )

    #check on test set
    from check_model_output import check_model_output

    testset_dice = check_model_output(save_path = save_path, 
                       model = model, 
                       dice_metric = dice_metric,
                       data_loader = test_loader,
                       device = device)

    ### Saving Data
    os.path.join(save_path,'info.txt')
    with open(os.path.join(save_path,'info.txt'),'w') as f:
        f.write("Test set mean dice:", testset_dice,
                "Number of train files:", number_of_training,
                "Number of val files:", number_of_validation,
                "model:", type(model),
                "loss function:", type(loss_function),
                "optimizer:", type(optimizer),
                "scheduler", type(scheduler),
                "metric:", type(dice_metric),
                "epochs:", number_of_epochs,
                )

    with open(os.path.join(save_path,'info.txt'),'a') as f:
        f.write("\nTrain transforms")
        for t in train_transforms.transforms:
             f.write(type(t))

        f.write("\nVal transforms")
        for t in val_transforms.transforms:
             f.write(type(t))

    with open(os.path.join(save_path,'losses.pkl'),'w') as f:
        pickle.dump([epoch_loss_values, metric_values], f)

    #Plot train & val loss 
    plt.figure("Train and Validation", (12, 6))
    plt.title("Epoch Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")   
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.plot(x, y)
    x = [2 * (i + 1) for i in range(len(metric_values))] #x = [val_interval * (i + 1) for i in range(len(metric_values))] #val_int =2
    y = 1-metric_values
    plt.plot(x, y)

    plt.savefig(os.path.join(save_path, "train_val_loss.png"), bbox_inches='tight')
    
    print("Finished!")


load_and_run(save_path="/home/omo23/Documents/segmentation-project/saved-tests/80-20-no-transform",
             tr_va_split = [80,20],
             number_of_epochs = 1000,)
