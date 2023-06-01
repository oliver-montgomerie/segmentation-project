from imports import *

###Data loading
#data_dir = "C:/Users/olive/OneDrive/Desktop/Liver Files"
data_dir = "/data/datasets/Liver/LiTS2017"

no_files = 131
train_images = sorted(glob.glob(os.path.join(data_dir, "Volumes", "*.nii")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "Segmentations", "*.nii")))
#train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii")))
#train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]


###Load data transforms
from choose_slice import SliceWithMaxNumLabelsd
from choose_slice import SaveSliced
save_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        SliceWithMaxNumLabelsd(["image", "label"], "label"),
        SaveSliced(["image", "label"],"/home/omo23/Documents/segmentation-project/2d-sliced-data")
    ]
)

ds = Dataset(data=data_dicts, transform=save_transforms)
data_loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=1) #no_files

data_dicts_single_slice = list(tqdm(dl_2d))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for batch_data in data_loader:
#     inputs, labels = (
#                 batch_data["image"],
#                 batch_data["label"],
#            )