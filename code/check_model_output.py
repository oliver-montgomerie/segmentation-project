from imports import *

def check_model_output(save_path, model, dice_metric, data_loader, device):
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

            #plot slices
            if i < 3:
                plt.figure("Comparison", (18, 6))
                plt.suptitle(dice_metric[-1])       #todo:check this will work?
                plt.subplot(1, 3, 1)
                plt.title(f"image")
                plt.imshow(test_inputs[i,0,:,:].detach().cpu(), cmap="gray")

                plt.subplot(1, 3, 2)
                plt.title(f"label")
                plt.imshow(test_labels[i,0,:,:].detach().cpu())

                plt.subplot(1, 3, 3)
                plt.title(f"prediction")
                plt.imshow(test_outputs[i,0,:,:].detach().cpu())

                fname = "img" + str(i) + ".png"
                plt.savefig(os.path.join(save_path, fname), bbox_inches='tight')

        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        print("Test set mean dice:", metric)
    return metric


data_dir = "/data/datasets/Liver/LiTS2017"
save_path = "/home/omo23/Documents/segmentation-project/saved-tests/" + "new-test/"

train_images = sorted(glob.glob(os.path.join(data_dir, "Volumes", "*.nii")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "Segmentations", "*.nii")))
#train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii")))
#train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
test_files = data_dicts[-3:]

from transforms import val_transforms
test_ds = Dataset(data=test_files, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=3, shuffle=True, num_workers=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dice_metric = DiceMetric(include_background=False, reduction="mean")
model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

testset_dice = check_model_output(save_path, model, dice_metric, test_loader, device)