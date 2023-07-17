from imports import *


def check_data():
    #Data loading

    data_dir = "/home/omo23/Documents/sliced-data"
    train_images = sorted(glob.glob(os.path.join(data_dir, "Images", "*.nii")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "Labels", "*.nii")))
    # data_dir = "C:/Users/olive/OneDrive/Desktop/Liver Files/sliced-data"   
    # train_images = sorted(glob.glob(os.path.join(data_dir, "Images", "*.nii")))
    # train_labels = sorted(glob.glob(os.path.join(data_dir, "Labels", "*.nii")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    check_files = data_dicts[:]

    #Load data transforms
    from transforms import load_slice_transforms

    ###datasets
    check_ds = Dataset(data=check_files, transform=load_slice_transforms)

    ###dataloaders
    check_loader = DataLoader(check_ds, batch_size=1, shuffle=False, num_workers=1)

    # don't need?
    region_counts = np.zeros(20)

    ###create empty write file
    with open('tumor_count.csv','w') as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(["file", "num-tumors", "num-tumor-pixels", "pixdim", "avg-size-mm"])

    for i, test_data in enumerate(check_loader):
        test_inputs, test_labels = (
            test_data["image"],
            test_data["label"],
        )
        fpath = test_data['image_meta_dict']['filename_or_obj'][0]
        fpath = fpath[fpath.rfind("/")+1:-4] 

        num_lbl_2 = np.sum(test_labels == 2).T.item()

        tumors = test_labels[0,0,:,:]
        tumors[tumors == 1] = 0
        seperated_labels, num_regions = seperate_instances(label_image = tumors, background=0, return_num=True, connectivity=None)

        pix_dim = test_inputs.pixdim[0][0].item()
        avg_size_mm = (num_lbl_2/num_regions)*pix_dim

        #with open('tumor_count.txt','a') as f:
        #    f.write(f"FILE: {fpath:3s} NUM_TUMOR: {num_regions:2.0f} TUMOR_PIX: {num_lbl_2:5.0f} AVG-TUMOR-SIZE: {(num_lbl_2/num_regions):5.1f} \n")

        with open("tumor_count.csv", "a", newline='',) as fp:
            writer = csv.writer(fp, delimiter=",")
            data = [fpath, str(num_regions), str(num_lbl_2), str(pix_dim), str(avg_size_mm)]
            writer.writerow(data)

        ### for viewing 
        # plt.figure("slice view", (12, 6))
        # plt.suptitle(fpath)
        # plt.subplot(1, 2, 1)
        # plt.title("image")
        # plt.imshow(test_inputs[0,0,:,:], cmap="gray")

        # plt.subplot(1, 2, 2)
        # plt.title("label")
        # plt.imshow(seperated_labels)
        # plt.show()
 
        

if __name__ == '__main__':
    check_data()