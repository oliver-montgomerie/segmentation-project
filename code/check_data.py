from imports import *

def check_data():
    #Data loading
    #data_dir = "/data/datasets/Liver/LiTS2017"   
    #train_images = sorted(glob.glob(os.path.join(data_dir, "Volumes", "*.nii")))
    #train_labels = sorted(glob.glob(os.path.join(data_dir, "Segmentations", "*.nii")))
    data_dir = "C:/Users/olive/OneDrive/Desktop/Liver Files"   
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    check_files = data_dicts[0:3]

    #Load data transforms
    from transforms import load_transforms

    ###datasets
    check_ds = Dataset(data=check_files, transform=load_transforms)

    ###dataloaders
    check_loader = DataLoader(check_ds, batch_size=1, shuffle=False, num_workers=1)

    for i, test_data in enumerate(check_loader):
        test_inputs, test_labels = (
            test_data["image"],
            test_data["label"],
        )
        fpath = test_data['image_meta_dict']['filename_or_obj'][0]
        fpath = fpath[-7:-4]
        if fpath[0] == "-": fpath = fpath[1:]

        num_lbl_2 = np.sum(test_labels == 2)

        tumors = test_labels[0,0,:,:]
        tumors[tumors == 1] = 0
        seperated_labels, num_regions = seperate_instances(label_image = tumors, background=0, return_num=True, connectivity=None)

        print("FILE:", fpath, "NUM_TUMOR:", num_regions, "TUMOR_PIX:", num_lbl_2, "AVG TUMOR SIZE:", round(num_lbl_2/num_regions,3))

        plt.figure("slice view", (12, 6))
        plt.suptitle(fpath)
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(test_inputs[0,0,:,:], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(seperated_labels)
        plt.show()

if __name__ == '__main__':
    check_data()