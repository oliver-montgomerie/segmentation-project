from imports import *
from choose_slice import SliceWithMaxNumLabelsd

class print_img_and_size(MapTransform):
    #for viewing 2d slices
    def __init__(self):
        pass

    def __call__(self, data):
        d = dict(data)
        print("Image shape", d['image'].shape)
        print("Label shape", d['label'].shape)

        plt.figure("check", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(d['image'], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(d['label'])
        #plt.show()
        
        return d

#for viewing how the transforms look
check_transforms = Compose(
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
        print_img_and_size(),

        RandAxisFlipd(keys=["image", "label"], prob=1),
        print_img_and_size(),

        # RandRotated(keys=["image", "label"], range_x=3.14, prob=1),
        # print_img_and_size(),

        RandZoomd(keys=["image", "label"], prob=0.1, min_zoom=0.9, max_zoom=1.1),
        print_img_and_size(),

        
    ]
)   

#todo: didnt add augmentations yet
augmented_train_transforms = Compose(
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
    ]
)

train_transforms = Compose(
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
        #Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        #print_img_size(),
        SliceWithMaxNumLabelsd(["image", "label"], "label"),
        #print_img_size(),

        #CropForegroundd(keys=["image", "label"], source_key="image"),
        #print_img_size(),
        #PadListDataCollate(),
        #print_img_size(),

        # user can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),
    ]
)

val_transforms = Compose(
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
        #Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        SliceWithMaxNumLabelsd(["image", "label"], "label"),
    ]
)

test_transforms = Compose(
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
        #Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        SliceWithMaxNumLabelsd(["image", "label"], "label"),
    ]
)