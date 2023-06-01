from imports import *

def check_model_output(model, save_path, data_loader, device):
    model_path = os.path.join(save_path, "best_metric_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            val_inputs, val_labels = (
                        batch["image"].to(device),
                        batch["label"].to(device),
                    )
            
            val_outputs = model(val_inputs)
            val_outputs = predict_segmentation(val_outputs, mutually_exclusive=True) # .detach().cpu()

            #plot slice
            for i in range(3):
                plt.figure("check", (18, 6))
                plt.subplot(1, 3, 1)
                plt.title(f"image")
                plt.imshow(val_inputs[i,0,:,:].detach().cpu(), cmap="gray")

                plt.subplot(1, 3, 2)
                plt.title(f"label")
                plt.imshow(val_labels[i,0,:,:].detach().cpu())

                plt.subplot(1, 3, 3)
                plt.title(f"output")
                plt.imshow(val_outputs[i,0,:,:].detach().cpu())

                fname = "img" + str(i) + ".png"
                plt.savefig(os.path.join(save_path, fname), bbox_inches='tight')


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
model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

check_model_output(model, save_path, test_loader, device)