from imports import *

def main():
    #print_config()

    #Data loading
    data_dir = "C:/Users/olive/OneDrive/Desktop/Liver Files"
    #data_dir = "/data/datasets/Liver/LiTS2017"

    no_files = 131
    #train_images = sorted(glob.glob(os.path.join(data_dir, "Volumes", "*.nii")))
    #train_labels = sorted(glob.glob(os.path.join(data_dir, "Segmentations", "*.nii")))
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    #train_files, val_files, test_files = data_dicts[0:3], data_dicts[3:6], data_dicts[-3:]
    train_files, val_files, test_files = data_dicts[0:(3*no_files//5)], data_dicts[(3*no_files//5):(4*no_files//5)], data_dicts[(4*no_files//5):]
    print("Number of train files:", len(train_files), "Number of val files:", len(val_files), "Number of test files:", len(test_files))

    #Load data transforms
    from transforms import train_transforms
    from transforms import val_transforms

    ### view sample of data
    #check_train_2d = Dataset(data = train_files, transform = train_transforms)
    #check_train_loader = DataLoader(check_train_2d, batch_size=1)
    #check_data = first(check_train_loader)
    #image, label = (check_data["image"][0][0], check_data["label"][0][0])
    #print(f"image shape: {image.shape}, label shape: {label.shape}")
    #plt.figure("check", (12, 6))
    #plt.subplot(1, 2, 1)
    #plt.title("image")
    #plt.imshow(image, cmap="gray")
    #plt.subplot(1, 2, 2)
    #plt.title("label")
    #plt.imshow(label)
    #plt.show()

    # #Load cache dataset ##doesnt work
    # try:
    #    with open('train_cache_files.pkl') as f: 
    #        train_ds = pickle.load(f)
    # except FileNotFoundError:
    #    print("train cached data not found. creating now")
    #    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=16)
    #    print("Cached data, now saving")
    #    with open('train_cache_files.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #        pickle.dump([train_ds], f)

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=16)
    #train_ds = Dataset(data=train_files, transform=train_transforms)

    # try: ##doesnt work
    #    with open('val_cache_files.pkl') as f: 
    #        val_ds = pickle.load(f)
    # except FileNotFoundError:
    #    print("val cached data not found. creating now")
    #    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=16)
    #    print("Cached data, now saving")
    #    with open('val_cache_files.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #        pickle.dump([val_ds], f)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=16)
    #val_ds = Dataset(data=val_files, transform=val_transforms)

    #dataloaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_ds, batch_size=16, num_workers=16)

    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
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
    plt.show()

if __name__ == "__main__":
    main()