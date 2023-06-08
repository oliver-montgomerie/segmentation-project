from imports import *
from choose_slice import SliceWithMaxNumLabelsd
        
load_transforms = Compose(
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
        Rotate90d(["image", "label"], k=1, spatial_axes=(0, 1)),
        Flipd(["image", "label"], spatial_axis=1),

    ]
)

deform = Rand2DElasticd(
    keys = ["image", "label"],
    prob=0.5,
    spacing=(5, 5),
    magnitude_range=(0,1),
    rotate_range=(np.pi / 20,),
    shear_range= (0.1,0.3),
    translate_range=(10, 10),
    scale_range=(-0.2, 0.2),
    padding_mode="zeros",
)


train_transforms = Compose(
    [
        load_transforms,
    ]
).flatten()

val_transforms = Compose(
    [
        load_transforms,
    ]
).flatten()

test_transforms = Compose(
    [
        load_transforms,
    ]
).flatten()

noise_train_transforms = Compose(
    [
        load_transforms,
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.2),
    ]
).flatten()

elastic_train_transforms = Compose(
    [
        load_transforms,
        deform,
    ]
).flatten()

noise_elastic_train_transforms = Compose(
    [
        load_transforms,
        deform,
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.2),
    ]
).flatten()




class print_img_and_size(MapTransform):
    #for viewing 2d slices
    def __init__(self, title):
        self.title = title

    def __call__(self, data):
        d = dict(data)
        image = d['image'][0,:,:]
        label = d['label'][0,:,:]
        print("Image shape:", d['image'].shape, "Label shape:", d['label'].shape)

        plt.figure("Transform data view", (12, 6))
        plt.suptitle(self.title)
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(image, cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(label)
        plt.show()
        
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
        Rotate90d(["image", "label"], k=1, spatial_axes=(0, 1)),
        Flipd(["image", "label"], spatial_axis=1),
        #print_img_and_size("True Slice"),

        # RandAxisFlipd(keys=["image", "label"], prob=1),
        # print_img_and_size("Axis flip"),

        # RandRotated(keys=["image", "label"], range_x=3.14, prob=1),
        # print_img_and_size("Rotated"),

        # RandZoomd(keys=["image", "label"], prob=0.1, min_zoom=0.75, max_zoom=1.25),
        # print_img_and_size("Zoom"),

        deform,
        #print_img_and_size("Elastic"),

        RandGaussianNoised(keys=["image"], prob=1, mean=0.0, std=0.1),
        print_img_and_size("Noise"),
        
    ]
)   


# train_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         ScaleIntensityRanged(
#             keys=["image"],
#             a_min=-200,
#             a_max=200,
#             b_min=0.0,
#             b_max=1.0,
#             clip=True,
#         ),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         #Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
#         #print_img_size(),
#         SliceWithMaxNumLabelsd(["image", "label"], "label"),
#         #print_img_size(),

#         #CropForegroundd(keys=["image", "label"], source_key="image"),
#         #print_img_size(),
#         #PadListDataCollate(),
#         #print_img_size(),

#         # user can also add other random transforms
#         # RandAffined(
#         #     keys=['image', 'label'],
#         #     mode=('bilinear', 'nearest'),
#         #     prob=1.0, spatial_size=(96, 96, 96),
#         #     rotate_range=(0, 0, np.pi/15),
#         #     scale_range=(0.1, 0.1, 0.1)),
#     ]
# )
