from imports import *
from add_tumor_transform import implant_tumor    

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


class flip_if_liver_on_right(MapTransform):
    #for viewing 2d slices
    def __init__(self, keys, label_key):
        self.keys = keys
        self.label_key = label_key

    def __call__(self, data):
        #get location of label == 1
        # if avg idx > width /2 then call flip
        d = dict(data)
        im = d['image']
        lbl = d[self.label_key]
        idx = np.argwhere(lbl>0) #location of liver
        mid_liver = np.mean(idx, axis = 0)

        if mid_liver[2] > lbl.shape[2]/2:
            lbl = torch.flip(lbl, [2])
            im = torch.flip(im, [2])
        d[self.label_key] = lbl
        d['image'] = im
        return d
    


load_slice_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Orientationd(keys=["image", "label"], axcodes="LA"),
        Rotate90d(["image", "label"], k=1, spatial_axes=(0, 1)),
        #flip_if_liver_on_right(keys=["image", "label"], label_key="label"),
        Spacingd(keys=["image", "label"], pixdim=(0.793, 0.793), mode=("bilinear", "nearest")),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size = [800,800]),
    ]
)


deform = Rand2DElasticd(
    keys = ["image", "label"],
    prob=0.5,
    spacing=(55, 55),
    magnitude_range=(-1.5,1.5),
    rotate_range=(np.pi / 20,),
    shear_range= (-0.05,0.05),
    translate_range=(-16, 16),
    scale_range=(-0.2, 0.2),
    padding_mode="zeros",
)


train_transforms = Compose(
    [
        load_slice_transforms,
    ]
).flatten()

val_transforms = Compose(
    [
        load_slice_transforms,
    ]
).flatten()

test_transforms = Compose(
    [
        load_slice_transforms,
    ]
).flatten()

noise_train_transforms = Compose(
    [
        load_slice_transforms,
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.2),
    ]
).flatten()

elastic_train_transforms = Compose(
    [
        load_slice_transforms,
        deform,
    ]
).flatten()

noise_elastic_train_transforms = Compose(
    [
        load_slice_transforms,
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
            #pickle.dump(label, f)
        
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

        num_rows = -(amount_of_plots // -3)     #this is like ceil(a/b)
        num_cols = -(amount_of_plots // -num_rows)

        for i, f in enumerate(files):
            with open(os.path.join(self.path, f), "rb") as input_file:
                img = pickle.load(input_file)
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(img, cmap="gray")
            plt.title(f[:-4])
            #plt.axis('off')

        plt.show()
        #shutil.rmtree(self.path)
        return d


deform_check = Rand2DElasticd(
    keys = ["image", "label"],
    prob=1,
    spacing=(55, 55),
    magnitude_range=(-1.5,1.5),
    rotate_range=(np.pi / 20,),
    shear_range= (-0.05,0.05),
    translate_range=(-10, 10),
    scale_range=(-0.2, 0.2),
    padding_mode="zeros",
)

#for viewing how the transforms look
check_temp_path = "/home/omo23/Documents/segmentation-project"
#check_temp_path = "C:/Users/olive/OneDrive/Desktop/Liver Files/segmentation-project"   
check_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        temps_save(1, check_temp_path),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        temps_save(2, check_temp_path),

        Orientationd(keys=["image", "label"], axcodes="LA"),
        Rotate90d(["image", "label"], k=1, spatial_axes=(0, 1)),
        flip_if_liver_on_right(keys=["image", "label"], label_key="label"),
        temps_save(3, check_temp_path),

        Spacingd(keys=["image", "label"], pixdim=(0.793, 0.793), mode=("bilinear", "nearest")),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size = [576,576]),
        temps_save(4, check_temp_path),

        deform_check,
        temps_save(6, check_temp_path),

        RandGaussianNoised(keys=["image"], prob=1, mean=0.0, std=0.2),
        temps_save(7, check_temp_path),
        plot_temps(check_temp_path),
    ]
).flatten() 

RT_train_transforms = Compose(  #rt real tumor
    [
        load_slice_transforms,
        implant_tumor(keys = ["image", "label"], t_type = "REAL", load_path = ""),
    ]
).flatten()

aug_RT_train_transforms = Compose(
    [
        load_slice_transforms,
        implant_tumor(keys = ["image", "label"], t_type = "REAL", load_path = ""),
        deform,
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.2),
    ]
).flatten()

VAE_train_transforms = Compose(
    [
        load_slice_transforms,
        implant_tumor(keys = ["image", "label"], t_type = "VAE", load_path = "/home/omo23/Documents/generative-models/VAE-models/latent-5-epochs-20"),

    ]
).flatten()

aug_VAE_train_transforms = Compose(
    [
        load_slice_transforms,
        implant_tumor(keys = ["image", "label"], t_type = "VAE", load_path = "/home/omo23/Documents/generative-models/VAE-models/latent-5-epochs-20"),
        deform,
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.2),
    ]
).flatten()

VAE_GAN_train_transforms = Compose(
    [
        load_slice_transforms,
        implant_tumor(keys = ["image", "label"], t_type = "VAE_GAN", load_path = "/home/omo23/Documents/generative-models/VAE-GAN-models/latent-5-epochs-30"),

    ]
).flatten()

aug_VAE_GAN_train_transforms = Compose(
    [
        load_slice_transforms,
        implant_tumor(keys = ["image", "label"], t_type = "VAE_GAN", load_path = "/home/omo23/Documents/generative-models/VAE-GAN-models/latent-5-epochs-30"),
        deform,
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.2),
    ]
).flatten()

