from imports import *


def view_images(batch_data, save_path):
    images, labels = (
        batch_data["image"],
        batch_data["label"],
    )
    for i in range(images.shape[0]): 
        plt.figure("Training data", (18, 6))
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.imshow(images[i,0,:,:].detach().cpu(), cmap="gray")

        plt.subplot(1,2,2)
        plt.axis('off')
        plt.imshow(labels[i,0,:,:].detach().cpu(), vmin =0, vmax =2)

        fpath = batch_data['image_meta_dict']['filename_or_obj'][i]
        fpath = fpath[fpath.rfind("/")+1:-4] 
        fname = "training-batch/img-" + fpath + ".png"
        # plt.savefig(os.path.join(save_path, fname), bbox_inches='tight')
        # plt.close()

        plt.suptitle(fpath)
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()

    return


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


    ###create empty write file
    with open('tumor_count.csv','w') as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(["file", "num-tumors", "num-tumor-pixels", "pixdim", "avg-size-mm"])

    oldfnum = -1
    for i, test_data in enumerate(check_loader):
        test_inputs, test_labels = (
            test_data["image"],
            test_data["label"],
        )
        fpath = test_data['image_meta_dict']['filename_or_obj'][0]
        fpath = fpath[fpath.rfind("/")+1:-4] 
        fnum = fpath[0:fpath.rfind("-")] 

        if fnum != oldfnum:
            oldfnum = fnum
            view_images(test_data, "")

        # num_lbl_2 = np.sum(test_labels == 2).T.item()

        # tumors = test_labels[0,0,:,:]
        # tumors[tumors == 1] = 0
        # seperated_labels, num_regions = seperate_instances(label_image = tumors, background=0, return_num=True, connectivity=None)

        # pix_dim = test_inputs.pixdim[0][0].item()
        # avg_size_mm = (num_lbl_2/num_regions)*pix_dim

        # #with open('tumor_count.txt','a') as f:
        # #    f.write(f"FILE: {fpath:3s} NUM_TUMOR: {num_regions:2.0f} TUMOR_PIX: {num_lbl_2:5.0f} AVG-TUMOR-SIZE: {(num_lbl_2/num_regions):5.1f} \n")

        # with open("tumor_count.csv", "a", newline='',) as fp:
        #     writer = csv.writer(fp, delimiter=",")
        #     data = [fpath, str(num_regions), str(num_lbl_2), str(pix_dim), str(avg_size_mm)]
        #     writer.writerow(data)

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