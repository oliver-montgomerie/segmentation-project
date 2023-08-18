from imports import *

def load_and_run(save_path = "", percentage_of_data = 100, number_of_epochs = 1000,
                 train_transforms = None, val_transforms = None,
                 use_all_slices = True):

    test_name = save_path[save_path.rfind("/")+1:] 
    csv_path = save_path[0:save_path.rfind("/")+1] + 'seg_results.csv'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, "test-comparisons"))
        os.makedirs(os.path.join(save_path, "training-batch"))
        print("created folder:",save_path)
    elif save_path[-4:] == "test":
        print("Re-writing test folder")
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, "test-comparisons"))
        os.makedirs(os.path.join(save_path, "training-batch"))
    else:
        print(save_path, " Folder already exists. Quitting...")
        return None

    num_workers = 4
    batch_size = 16
    learning_rate = 1e-3
    scheduler_gamma = 0.9
    scheduler_step_size = 5

    #Data loading
    data_dir = "/home/omo23/Documents/sliced-data"
    all_images = sorted(glob.glob(os.path.join(data_dir, "Images", "*.nii")))
    all_labels = sorted(glob.glob(os.path.join(data_dir, "Labels", "*.nii")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)]
    
    run_train_files_nums = train_files_nums[0:round( len(train_files_nums) * (percentage_of_data/100) ) ]

    test_files, val_files, train_files = [], [], []
    for d in data_dicts:
        d_num = d['image']
        d_num = d_num[d_num.rfind("/")+1:d_num.rfind("-")] 
        if d_num in test_files_nums:
            test_files.append(d)
        if d_num in val_files_nums:
            val_files.append(d)
        if d_num in run_train_files_nums: 
            train_files.append(d)

    # filter out slices with small tumor area.
    if not use_all_slices: train_files = [item for item in train_files if file_tumor_size(item) > min_tumor_size]
    #else: train_files = [item for item in train_files if (file_tumor_size(item) > min_tumor_size) or (file_tumor_size(item) == 0)]

    val_files  = [item for item in val_files if file_tumor_size(item) > min_tumor_size]
    test_files = [item for item in test_files if file_tumor_size(item) > min_tumor_size]
    
    print("Number of train files:", len(run_train_files_nums), "Number of val files:", len(val_files_nums), "Number of test files:", len(test_files_nums))
    print("Number of train slices:", len(train_files), "Number of val slices:", len(val_files), "Number of test slices:", len(test_files))

    ###datasets
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=num_workers) # 0.5
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)
    # train_ds = Dataset(data=train_files, transform=train_transforms)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    
    ###dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers) 

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
    
    loss_function = DiceLoss(softmax=True) #to_onehot_y=True, 
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = StepLR(optimizer, step_size = scheduler_step_size, gamma=scheduler_gamma)     #ReduceLROnPlateau 
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    print("Created model, loss, optim ,dice")

    ### Training loop
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
                    save_path = save_path,
                    max_epochs = number_of_epochs,
                    )
    
    ###Plot train & val loss 
    with open(os.path.join(save_path,'losses.pkl'),'wb') as f:
        pickle.dump([epoch_loss_values, metric_values], f)

    plt.figure("Train and Validation", (12, 6))
    plt.title("Epoch Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")   
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.plot(x, y, label="train")

    x = [1 * (i + 1) for i in range(len(metric_values))] #x = [val_interval * (i + 1) for i in range(len(metric_values))] #val_int =2
    y = list(map(lambda x:1-x, metric_values))
    plt.plot(x, y, label="val")

    plt.legend(loc="upper right")
    plt.savefig(os.path.join(save_path, "train_val_loss.png"), bbox_inches='tight')
    plt.close()
    

    ### check on test set
    num_workers = 4
    batch_size = 16
    from check_model_output import check_model_output
    from transforms import test_transforms
    test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=num_workers)
    # test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    testset_dice_metric = DiceMetric(include_background=False, reduction="none")

    testset_dice = check_model_output(save_path = save_path, 
                       model = model, 
                       dice_metric = testset_dice_metric,
                       data_loader = test_loader,
                       device = device,
                       num_test_files = len(test_files))

    ### Saving Data
    os.path.join(save_path,'info.txt')
    with open(os.path.join(save_path,'info.txt'),'w') as f:
        txt = ["Test set mean liver dice:", round(testset_dice[0],3),
               "\nTest set mean tumor dice:", round(testset_dice[1],3),
                "\nNumber of train files:", len(run_train_files_nums), " Number of train slices:", len(train_files), 
                "\nNumber of val files:", len(val_files_nums), " Number of val slices:", len(val_files), 
                "\nNumber of test files:", len(test_files_nums), " Number of test slices:", len(test_files), 
                "\nbatch size:", batch_size,
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
                ", scheduler step size:", scheduler_step_size,
                "\nmetric:", type(dice_metric),
                "\nepochs:", number_of_epochs,
        ]
        for t in txt:
            f.write(f"{t}")

    with open(csv_path, "a", newline='',) as fp:
        writer = csv.writer(fp, delimiter=",")
        data = [test_name, percentage_of_data, round(testset_dice[0],3), round(testset_dice[1],3)]
        writer.writerow(data)

    with open(os.path.join(save_path,'info.txt'),'a') as f:
        f.write("\nTrain transforms")
        for t in train_transforms.transforms:
             f.write("\n"+str(type(t)))

        f.write("\nVal transforms")
        for t in val_transforms.transforms:
             f.write("\n"+str(type(t)))

    
        
    print("Finished!")


