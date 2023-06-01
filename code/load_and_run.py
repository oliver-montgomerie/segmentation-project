from imports import *

#todo: make txt save data in the folder
# add batch size and epochs along with the #train /#test

def load_and_run(tr_va_te_split=[80,10,10],
                 save_path = "",
                 ):
    num_workers = 16
    #Data loading
    #data_dir = "C:/Users/olive/OneDrive/Desktop/Liver Files"
    data_dir = "/data/datasets/Liver/LiTS2017"

    no_files = 131
    number_of_training = (tr_va_te_split[0] * no_files) // 100
    number_of_validation = (tr_va_te_split[1] * no_files) // 100
    number_of_test = no_files - number_of_training - number_of_validation
    
    train_images = sorted(glob.glob(os.path.join(data_dir, "Volumes", "*.nii")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "Segmentations", "*.nii")))
    #train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii")))
    #train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    train_files = data_dicts[0:number_of_training]
    val_files = data_dicts[number_of_training:number_of_validation]
    test_files = data_dicts[-number_of_test:]
    print("Number of train files:", len(train_files), "Number of val files:", len(val_files), "Number of test files:", len(test_files))

    #Load data transforms
    from transforms import train_transforms
    from transforms import val_transforms

    ###datasets
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=num_workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)
    test_ds = Dataset(data=test_files, transform=val_transforms)

    ###dataloaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=16, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=3, shuffle=True, num_workers=num_workers)

    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
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
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    print("Created model, loss, optim ,dice")

    #Training loop
    from training_loop import training_loop
    epoch_loss_values, metric_values = training_loop(
                    model = model,
                    train_loader = train_loader,
                    val_loader = val_loader,
                    optimizer = optimizer,
                    loss_function = loss_function,
                    dice_metric = dice_metric,
                    device = device,
                    max_epochs = 500,
                    model_path_and_name = os.path.join(save_path, "best_metric_model.pth"),
                    )

    #Plots
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [2 * (i + 1) for i in range(len(metric_values))] #x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)

    plt.savefig(os.path.join(save_path, "train_val_loss.png"), bbox_inches='tight')
    
    from check_model_output import check_model_output

    check_model_output(model = model, 
                       model_path = save_path, 
                       data_loader = test_loader,
                       device = device)


load_and_run([80,10,10],"/home/omo23/Documents/segmentation-project/saved-tests/80-10-10")
load_and_run([70,20,10],"/home/omo23/Documents/segmentation-project/saved-tests/80-10-10")
load_and_run([60,30,10],"/home/omo23/Documents/segmentation-project/saved-tests/80-10-10")
load_and_run([50,40,10],"/home/omo23/Documents/segmentation-project/saved-tests/80-10-10")
load_and_run([40,50,10],"/home/omo23/Documents/segmentation-project/saved-tests/80-10-10")
load_and_run([30,60,10],"/home/omo23/Documents/segmentation-project/saved-tests/80-10-10")
load_and_run([20,70,10],"/home/omo23/Documents/segmentation-project/saved-tests/80-10-10")
load_and_run([10,80,10],"/home/omo23/Documents/segmentation-project/saved-tests/80-10-10")