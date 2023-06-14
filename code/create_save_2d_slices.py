from imports import *
#doesnt work 
#tried using monai save transform but get som errors


def stuff():
    #Data loading
    data_dir = "/data/datasets/Liver/LiTS2017"   
    train_images = sorted(glob.glob(os.path.join(data_dir, "Volumes", "*.nii")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "Segmentations", "*.nii")))
    # data_dir = "C:/Users/olive/OneDrive/Desktop/Liver Files"   
    # train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii")))
    # train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    check_files = data_dicts[0:1]

    #Load data transforms
    from transforms import load_transforms

    save_transforms = Compose(
        [
            load_transforms,
            
        ]
    ).flatten()
    
    ###datasets
    check_ds = CacheDataset(data=check_files, transform=save_transforms, cache_rate=1.0, num_workers=16) 

    ###dataloaders
    check_loader = DataLoader(check_ds, batch_size=16, shuffle=False, num_workers=16)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i, test_data in enumerate(check_loader):
        test_inputs, test_labels = (
                    test_data["image"].to(device),
                    test_data["label"].to(device),
                )
        print(f"{i}/{len(check_loader)}")

        # path = "/home/omo23/Documents/sliced-data"
        # nib.save(img = , filename = )


if __name__ == '__main__':
    stuff()