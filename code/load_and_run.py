from imports import *

def load_and_run(save_path = "", tr_va_split=[60,20,20], number_of_epochs = 1000,
                 train_transforms = None, val_transforms = None):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, "test-comparisons"))
        print("created folder:",save_path)
    else:
        print(save_path, " Folder already exists. Quitting...")
        quit()

    num_workers = 16
    train_batch_size = 16
    learning_rate = 1e-3
    scheduler_gamma = 0.9

    #Data loading
    data_dir = "/data/datasets/Liver/LiTS2017"   
    train_images = sorted(glob.glob(os.path.join(data_dir, "Volumes", "*.nii")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "Segmentations", "*.nii")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    no_files = len(data_dicts) 
    number_of_test = (tr_va_split[2] * no_files) // 100
    number_of_validation = (tr_va_split[1] * no_files) // 100
    number_of_training = no_files - number_of_test - number_of_validation
    
    test_files = data_dicts[-number_of_test:]
    val_files = data_dicts[-(number_of_test+number_of_validation):-number_of_test]
    train_files = data_dicts[0:-(number_of_test+number_of_validation)]

    # train_files = data_dicts[0:1] 
    # val_files = data_dicts[1:2]
    # test_files = data_dicts[2:5]
    
    print("Number of train files:", len(train_files), "Number of val files:", len(val_files), "Number of test files:", len(test_files))

    #Load data transforms
    #from transforms import train_transforms, val_transforms, test_transforms
    from transforms import test_transforms

    ###datasets
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=num_workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)
    test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=num_workers)
    # train_ds = Dataset(data=train_files, transform=train_transforms)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    # test_ds = Dataset(data=test_files, transform=test_transforms)

    ###dataloaders
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers) #train_batch_size
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=True, num_workers=num_workers) 
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, num_workers=num_workers)

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
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    #scheduler = ExponentialLR(optimizer, gamma=scheduler_gamma)     #ReduceLROnPlateau
    scheduler = StepLR(optimizer, step_size = 25, gamma=scheduler_gamma)     
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
                       device = device,
                       num_test_files = len(test_files))

    ### Saving Data
    os.path.join(save_path,'info.txt')
    with open(os.path.join(save_path,'info.txt'),'w') as f:
        txt = ["Test set mean dice:", round(testset_dice,3),
                "\nNumber of train files:", number_of_training,
                "\nNumber of val files:", number_of_validation,
                "\nNumber of test files:", number_of_test,
                "\nTrain batch size:", train_batch_size,
                "\nmodel:", type(model),
                "\n   channels:", model.channels,
                "\n   strides:", model.strides,
                "\n   res units:", model.num_res_units,
                "\n   norm:", model.norm,
                "\nloss function:", type(loss_function),
                "\noptimizer:", type(optimizer),
                "\nlearning rate:", learning_rate,
                "\nscheduler", type(scheduler),
                ", scheduler gamma:", scheduler_gamma,
                "\nmetric:", type(dice_metric),
                "\nepochs:", number_of_epochs,
        ]
        for t in txt:
            f.write(f"{t}")

    with open(os.path.join(save_path,'info.txt'),'a') as f:
        f.write("\nTrain transforms")
        for t in train_transforms.transforms:
             f.write("\n"+str(type(t)))

        f.write("\nVal transforms")
        for t in val_transforms.transforms:
             f.write("\n"+str(type(t)))

    with open(os.path.join(save_path,'losses.pkl'),'wb') as f:
        pickle.dump([epoch_loss_values, metric_values], f)

    #Plot train & val loss 
    plt.figure("Train and Validation", (12, 6))
    plt.title("Epoch Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")   
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.plot(x, y, label="train")

    x = [2 * (i + 1) for i in range(len(metric_values))] #x = [val_interval * (i + 1) for i in range(len(metric_values))] #val_int =2
    y = list(map(lambda x:1-x, metric_values))
    plt.plot(x, y, label="val")

    plt.legend(loc="upper right")
    plt.savefig(os.path.join(save_path, "train_val_loss.png"), bbox_inches='tight')
    
    print("Finished!")


