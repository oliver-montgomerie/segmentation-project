from imports import *
#needs debugging

#Data loading
data_dir = "/data/datasets/Liver/LiTS2017"   
train_images = sorted(glob.glob(os.path.join(data_dir, "Volumes", "*.nii")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "Segmentations", "*.nii")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

check_files = data_dicts[0:1]

#Load data transforms
from transforms import check_transforms 

for t in check_transforms.transforms:
    print(str(type(t)))

###datasets
# train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=num_workers)
# val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)
check_ds = Dataset(data=check_files, transform=check_transforms)

###dataloaders
check_loader = DataLoader(check_ds, batch_size=1, shuffle=False, num_workers=1)
check_data = first(check_loader)
check_data = first(check_loader)
check_data = first(check_loader)
plt.show()


