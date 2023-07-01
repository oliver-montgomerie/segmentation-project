from imports import *
#needs debugging

def stuff():
    #Data loading
    data_dir = "/home/omo23/Documents/sliced-data"
    all_images = sorted(glob.glob(os.path.join(data_dir, "Images", "*.nii")))
    all_labels = sorted(glob.glob(os.path.join(data_dir, "Labels", "*.nii")))
    # data_dir = "C:/Users/olive/OneDrive/Desktop/Liver Files/test-data"   
    # train_images = sorted(glob.glob(os.path.join(data_dir, "Images", "*.nii")))
    # train_labels = sorted(glob.glob(os.path.join(data_dir, "Labels", "*.nii")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)]

    check_files = data_dicts[0:1]
    # fs = np.random.randint(0, len(data_dicts), 10)
    # for f in fs:
    #     check_files.append(data_dicts[f]) 
    # print(check_files)

    add_90 = False
    for d in data_dicts:
        d_num = d['image']
        d_num = d_num[d_num.rfind("/")+1:d_num.rfind("-")]
        if d_num == "40" and not add_90:
            check_files.append(d)
            add_90 = True

    #Load data transforms
    from transforms import check_transforms, load_slice_transforms

    for t in check_transforms.transforms:
        print(str(type(t)))

    ###datasets
    check_ds = Dataset(data=check_files, transform=check_transforms)

    ###dataloaders
    check_loader = DataLoader(check_ds, batch_size=1, shuffle=False, num_workers=1) #,collate_fn=PadListDataCollate()
    #check_data = first(check_loader)
    for i, test_data in enumerate(check_loader):
        print(i)
    #check_data = first(check_loader)
    #check_data = first(check_loader)
    # check_data = first(check_loader)
    # check_data = first(check_loader)


if __name__ == '__main__':
    stuff()