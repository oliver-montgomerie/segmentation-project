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
        SliceWithMaxNumLabelsd(["image", "label"], "label"),
        Rotate90d(["image", "label"], k=1, spatial_axes=(0, 1)),
        Flipd(["image", "label"], spatial_axis=1),
        Spacingd(keys=["image", "label"], pixdim=(0.556641, 0.556641), mode=("bilinear", "nearest")),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size = [800,800]),

    ]
)

deform = Rand2DElasticd(
    keys = ["image", "label"],
    prob=0.5,
    spacing=(30, 30),
    magnitude_range=(1.5,1.5),
    rotate_range=(np.pi / 20,),
    shear_range= (-0.05,0.05),
    translate_range=(-10, 10),
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


class temps_save(MapTransform):
    #for viewing 2d slices
    def __init__(self, title, path):
        self.title = str(title) + ".pkl"
        self.path = os.path.join(path, 'temp')

    def __call__(self, data):
        d = dict(data)
        image = d['image'][0,:,:]
        label = d['label'][0,:,:]
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        else:
            pass #folder exists

        with open(os.path.join(self.path, self.title),'wb') as f:
            pickle.dump(image, f)
        
        return d
    
    
class plot_temps(MapTransform):
    #for viewing 2d slices
    def __init__(self, path):
        self.path = os.path.join(path, 'temp')

    def __call__(self, data):
        d = dict(data)
        files = os.listdir(self.path)
        amount_of_plots = len(files)
        plt.figure("Data loader steps", (12, 6))
        plt.suptitle("Data loader steps")

        for i, f in enumerate(files):
            with open(os.path.join(self.path, f), "rb") as input_file:
                img = pickle.load(input_file)
            plt.subplot(1, amount_of_plots, i+1)
            plt.imshow(img, cmap="gray")
            plt.title(f[:-4])
            #plt.axis('off')

        plt.show()
        #shutil.rmtree(self.path)
        return d


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
        plt.axis('off')
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
        SliceWithMaxNumLabelsd(["image", "label"], "label"),        #get 2d
        temps_save(1, "/home/omo23/Documents/segmentation-project"),

        Rotate90d(["image", "label"], k=1, spatial_axes=(0, 1)),
        temps_save(2, "/home/omo23/Documents/segmentation-project"),

        Flipd(["image", "label"], spatial_axis=1),
        temps_save(3, "/home/omo23/Documents/segmentation-project"),
        #print_img_and_size("original slice"),

        Spacingd(keys=["image", "label"], pixdim=(0.556641, 0.556641), mode=("bilinear", "nearest")),
        #print_img_and_size("resampled slice"),
        temps_save(4, "/home/omo23/Documents/segmentation-project"),
        

        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size = [800,800]),
        temps_save(5, "/home/omo23/Documents/segmentation-project"),

        # deform,
        # RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
        # print_img_and_size("Elastic + Noise"),
        plot_temps("/home/omo23/Documents/segmentation-project"),
    ]
)   
