from imports import *

def check_model_output(save_path, model, dice_metric, data_loader, device, num_test_files):
    model_path = os.path.join(save_path, "best_metric_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for i, test_data in enumerate(data_loader):
            test_inputs, test_labels = (
                        test_data["image"].to(device),
                        test_data["label"].to(device),
                    )
            
            test_outputs = model(test_inputs)
            test_outputs = predict_segmentation(test_outputs, mutually_exclusive=True) # .detach().cpu()

            one_hot_out = one_hot(test_outputs, num_classes=3, dim=1)

            one_hot_labels = one_hot(test_labels, num_classes=3, dim=1)
            dice_metric(y_pred=one_hot_out, y=one_hot_labels)
            x = dice_metric.get_buffer()

            #plot slices
            if i == 0: #should just be 1 batch. batch size = number of test data
                for j in range(num_test_files):
                    txt = "Liver dice: " + str(round(x[j][0].item(),3)) + ", Tumour dice: "+ str(round(x[j][1].item(),3))
                    plt.figure("Comparison", (18, 6))
                    plt.suptitle(txt)       
                    plt.subplot(1, 3, 1)
                    plt.title(f"image")
                    plt.imshow(test_inputs[j,0,:,:].detach().cpu(), cmap="gray")

                    plt.subplot(1, 3, 2)
                    plt.title(f"label")
                    plt.imshow(test_labels[j,0,:,:].detach().cpu())

                    plt.subplot(1, 3, 3)
                    plt.title(f"prediction")
                    plt.imshow(test_outputs[j,0,:,:].detach().cpu())

                    fname = "test-comparisons/img" + str(j) + ".png"
                    plt.savefig(os.path.join(save_path, fname), bbox_inches='tight')
                    plt.close()

        metric = dice_metric.aggregate(reduction="mean_batch")
        #metric = dice_metric.aggregate(reduction="mean_channel").item()
        dice_metric.reset()
        #print(f"Test set mean dice: {metric:.4f}")
        print(f"Test set liver mean dice: {metric[0].item():.4f}")
        print(f"Test set tumour mean dice: {metric[1].item():.4f}")
    return [metric[0].item(), metric[1].item()]


# data_dir = "/data/datasets/Liver/LiTS2017"
# save_path = "/home/omo23/Documents/segmentation-project/saved-tests/" + "new-test/"

# train_images = sorted(glob.glob(os.path.join(data_dir, "Volumes", "*.nii")))
# train_labels = sorted(glob.glob(os.path.join(data_dir, "Segmentations", "*.nii")))
# #train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii")))
# #train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii")))
# data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
# test_files = data_dicts[-3:]

# from transforms import val_transforms
# test_ds = Dataset(data=test_files, transform=val_transforms)
# test_loader = DataLoader(test_ds, batch_size=3, shuffle=True, num_workers=1)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dice_metric = DiceMetric(include_background=False, reduction="mean")
# model = UNet(
#         spatial_dims=2,
#         in_channels=1,
#         out_channels=3,
#         channels=(16, 32, 64, 128, 256),
#         strides=(2, 2, 2, 2),
#         num_res_units=2,
#         norm=Norm.BATCH,
#     ).to(device)

# testset_dice = check_model_output(save_path, model, dice_metric, test_loader, device)