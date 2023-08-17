from imports import *

tumor_deform = Rand2DElasticd(
    keys = ["im"],
    prob=1,
    spacing=(55, 55),
    magnitude_range=(-1.1,1.1),
    rotate_range=(np.pi,),
    #shear_range= (-0.01,0.01),
    scale_range=(-0.3, 0.1),
    padding_mode="zeros",
)

load_tumor_transforms = Compose(
    [
        LoadImaged(keys=["im"], image_only=False),
        EnsureChannelFirstd(keys=["im"]),
        ScaleIntensityRanged(keys=["im"],
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True,),
        Orientationd(keys=["im"], axcodes="LA"),
        Spacingd(keys=["im"], pixdim=(0.793, 0.793), mode=("bilinear")),
        ResizeWithPadOrCropd(keys=["im"], spatial_size = [640,640]),
        tumor_deform,
    ]
).flatten() 

def generate_a_tumor(model, latent_size, dist, tumor_shape): #, device):
    with torch.no_grad():
        sample = torch.zeros(1, latent_size) #.to(device)
        for s in range(sample.shape[1]):
            #maybe This should be weighted towards the edges for unusual looking tumors?
            sample[0,s] = dist.icdf((torch.rand(1)*0.9) + 0.05)

        o = model.decode_forward(sample) #get output from latent sample
        #o = o.detach().cpu().numpy().reshape(tumor_shape)
        #o = torch.reshape(o, )

        mask = torch.zeros(o.shape)    #create mask from thresholding tumor
        thresh = (torch.max(o) + torch.min(o))/2
        mask[o > thresh] = 1

        #clip the tumor to the mask size ... or not'. let the distance map do this part
        #o[mask != 1] = 0

        # # make sure its large enough
        # tumor_size = np.sum(mask == 1) * 0.793 * 0.793
        # tumor_size = int(tumor_size)
        # if tumor_size < min_tumor_size:
        #     continue

        return o, mask #o, mask  
    
# def get_real_tumor(tumor_dl): 
#     #load a random tumour
#     got_tumor = False
    
#     while not got_tumor:

#         tumor = next(iter(tumor_dl))
#         tumor = tumor['im']

#         fnum = tumor.meta['filename_or_obj'][0]
#         fnum = fnum[fnum.rfind("/")+1:fnum.rfind("-")]

#         if fnum in train_files_nums:
#             got_tumor = True

#             np_mask = torch.zeros(tumor.shape)
#             thresh = 0.15 #(torch.max(tumor) + torch.min(tumor))/3
#             np_mask[tumor > thresh] = 1

#             # print(tumor.pixdim[0])
#             # t_size = torch.count_nonzero(np_mask) * tumor.pixdim[0][0] * tumor.pixdim[0][1] 
#             # if t_size < min_tumor_size:
#             #     got_tumor = False
   
#     return tumor, np_mask

def get_real_tumor(): 
    #load a random tumour
    got_tumor = False
    folder = "/home/omo23/Documents/tumor-patches-data/Images/"
    
    while not got_tumor:
        file = random.choice(os.listdir(folder))
        fnum = file[:file.rfind("-")] 
    
        if fnum in train_files_nums:
            got_tumor = True

            load_d = {"im": os.path.join(folder, file)}
            tumor = load_tumor_transforms(load_d)
            tumor = tumor['im']

            np_mask = torch.zeros(tumor.shape)
            thresh = 0.15 #(torch.max(tumor) + torch.min(tumor))/3
            np_mask[tumor > thresh] = 1

            # print(tumor.pixdim[0])
            # t_size = torch.count_nonzero(np_mask) * tumor.pixdim[0][0] * tumor.pixdim[0][1] 
            # if t_size < min_tumor_size:
            #     got_tumor = False
   
    return tumor, np_mask



def add_tumor_to_slice(img, lbl, t_type, model, latent_size, tumor_shape): #, device):
    dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))

    liver_pix = torch.argwhere(lbl == 1)

    max_attempts = 20
    good_placement = False
    for attempt in range(max_attempts):
        if t_type == "REAL":
            tumor_img, tumor_lbl = get_real_tumor() #could pass train file nums?
            #tumor_img = tumor_img[0,:,:,:]
            #tumor_lbl = tumor_lbl[0,:,:,:].int()
        else:
            tumor_img, tumor_lbl = generate_a_tumor(model, latent_size, dist, tumor_shape) #, device)
            tumor_img = tumor_img[0,:,:,:]
            tumor_lbl = tumor_lbl[0,:,:,:].int()

        # plt.figure("New tumour", (18, 6))
        # plt.subplot(1,2,1)
        # plt.imshow(tumor_img.detach().cpu().numpy()[0,:, :], cmap="gray", vmin=0, vmax=1)
        # plt.axis('off')

        # plt.subplot(1,2,2)
        # plt.imshow(tumor_lbl.detach().cpu().numpy()[0,:, :], cmap="gray", vmin=0, vmax=1)
        # plt.axis('off')

        # plt.show()
        # plt.pause(1)


        # pad to slice size
        pad_size = [img.size(dim=1) - tumor_img.size(dim=1), img.size(dim=2) - tumor_img.size(dim=2)]
        tumor_img = F.pad(tumor_img, (0, pad_size[0], 0, pad_size[1]), "constant", torch.min(tumor_img))

        pad_size = [lbl.size(dim=1) - tumor_lbl.size(dim=1), lbl.size(dim=2) - tumor_lbl.size(dim=2)]
        tumor_lbl = F.pad(tumor_lbl, (0, pad_size[0], 0, pad_size[1]), "constant", 0)
        
        #choose location to insert into liver
        tumor_pix = torch.argwhere(tumor_lbl == 1).to(torch.float)
        tumor_centre = torch.round(torch.mean(tumor_pix, axis = 0)).to(torch.int)

        location = random.choice(liver_pix).to(torch.int)
        a = (location[1] - tumor_centre[1]).item()
        b = (location[2] - tumor_centre[2]).item()

        #shift the tumor so that the centre = location.
        tumor_lbl = torch.roll(tumor_lbl, a, dims=1)
        tumor_lbl = torch.roll(tumor_lbl, b, dims=2)
        tumor_img = torch.roll(tumor_img, a, dims=1)
        tumor_img = torch.roll(tumor_img, b, dims=2)

        tumor_lbl[tumor_lbl==1] = 3
        # add tumor label to img label. if there are lots of 3's then it means outside liver
        # if lots of 4's then it was inside the liver
        # 5 means on- top of another tumor, which is also ok
        gen_lbl = lbl + tumor_lbl

        # todo: check if this returns the right length or *2 because 2d
        not_liver = len(torch.argwhere(gen_lbl == 3))
        in_liver = len(torch.argwhere(gen_lbl > 3))

        #print(f"in_liver ratio: {in_liver / (in_liver + not_liver)}")
        if in_liver / (in_liver + not_liver) > 0.95:
            good_placement = True
            break #good

        # Possible Todo: add a probability for repeating the process for adding multiple tumors.

    if good_placement == False:
        return img, lbl 
        
    ## merge images
    #todo: sort this shit

    sobel_h = torch.from_numpy(ndimage.sobel(tumor_lbl[0,:,:], 0) ) # horizontal gradient
    sobel_v = torch.from_numpy(ndimage.sobel(tumor_lbl[0,:,:], 1) ) # vertical gradient
    tumor_edges = torch.sqrt(sobel_h**2 + sobel_v**2)
    tumor_edges = tumor_edges / torch.max(tumor_edges)

    edge_locations = torch.argwhere(tumor_edges > 0.5).to(torch.float)
    lbl_locations = torch.argwhere(tumor_lbl[0,:,:] >= 3).to(torch.float)

    dists, _ = torch.min(torch.cdist(lbl_locations, edge_locations), dim=1)

    distmap = torch.zeros(list(tumor_lbl.size()))
    distmap[tumor_lbl >= 3] = torch.add(dists, 1)

    #distmap[distmap < 3] = distmap[distmap < 3] / (1.2*torch.max(distmap) )
    distmap[distmap < 3] = tumor_img[distmap < 3]
    distmap[distmap >= 3] = 1
    distmap = torch.square(distmap)

    gen_img = img.detach().clone() 

    #gen_img[tumor_lbl >= 3] = tumor_img[tumor_lbl >= 3] #(0.8*tumor_img[tumor_lbl >= 3]) + (0.2*gen_img[tumor_lbl >= 3])
    gen_img[tumor_lbl >= 3] = ((distmap*tumor_img)[tumor_lbl >= 3]) + ((1-distmap)*img)[tumor_lbl >= 3]


    # ### plots
    buffer = 5
    rows = torch.any(tumor_lbl, axis=2)
    cols = torch.any(tumor_lbl, axis=1)
    ymin, ymax = torch.where(rows)[1][[0, -1]] 
    xmin, xmax = torch.where(cols)[1][[0, -1]] 
    ymin = ymin - buffer
    ymax = ymax + buffer
    xmin = xmin - buffer
    xmax = xmax + buffer

    plt.figure("New tumour", (18, 6))
    plt.subplot(1,2,1)
    plt.imshow(tumor_img.detach().cpu().numpy()[0,ymin:ymax, xmin:xmax], cmap="gray", vmin=0, vmax=1)
    plt.title("Tumour")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(distmap.detach().cpu().numpy()[0,ymin:ymax, xmin:xmax], cmap="gray", vmin=0, vmax=1)
    plt.title("Distance Map")
    plt.axis('off')

    plt.show()
    plt.pause(1)

    buffer = 5
    rows = torch.any(lbl, axis=2)
    cols = torch.any(lbl, axis=1)
    ymin, ymax = torch.where(rows)[1][[0, -1]] 
    xmin, xmax = torch.where(cols)[1][[0, -1]] 
    ymin = ymin - buffer
    ymax = ymax + buffer
    xmin = xmin - buffer
    xmax = xmax + buffer

    plt.figure("New tumor", (18, 6))
    plt.subplot(2,3,1)
    plt.imshow(img.detach().cpu().numpy()[0,ymin:ymax, xmin:xmax], cmap="gray", vmin=0, vmax=1)
    plt.title("Original")
    plt.axis('off')
    plt.subplot(2,3,4)
    plt.imshow(lbl.detach().cpu().numpy()[0,ymin:ymax, xmin:xmax], vmin=0, vmax=5)
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.imshow(tumor_img.detach().cpu().numpy()[0,ymin:ymax, xmin:xmax], cmap="gray", vmin=0, vmax=1)
    plt.axis('off')
    plt.title("New tumor")
    plt.subplot(2,3,5)
    plt.imshow(tumor_lbl.detach().cpu().numpy()[0,ymin:ymax, xmin:xmax], vmin=0, vmax=5)
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.imshow(gen_img.detach().cpu().numpy()[0,ymin:ymax, xmin:xmax], cmap="gray", vmin=0, vmax=1)
    plt.title("Implanted tumor")
    plt.axis('off')
    plt.subplot(2,3,6)
    plt.imshow(gen_lbl.detach().cpu().numpy()[0,ymin:ymax, xmin:xmax], vmin=0, vmax=5)
    plt.axis('off')

    plt.show()
    plt.pause(1)

    gen_lbl[gen_lbl >= 3] = 2 

    return gen_img, gen_lbl



class implant_tumor(MapTransform):
    def __init__(self, keys, t_type, load_path):
        #model shape should definetly be saved in a config somewhere instead of this...
        self.keys = keys

        self.t_type = t_type #REAL, VAE, VAE-GAN, DIFF... 

        # if self.t_type == "REAL":
        #     self.tumor_dl = tumor_dl
        # else:
        #     self.tumor_dl = None


        self.tumor_shape = [1,256,256]
        self.latent_size = 5
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VarAutoEncoder(
            spatial_dims=2,
            in_shape=self.tumor_shape,
            out_channels=1,
            latent_size=self.latent_size,
            channels=(16, 32, 64, 128, 256),
            strides=(1, 2, 2, 2, 2),
        )#.to(self.device)
        
        if t_type == "VAE" or t_type == "VAE_GAN":
            self.model.load_state_dict(torch.load(os.path.join(load_path, "trained_model.pth")))
            self.model.eval()

    def __call__(self, data):
        d = dict(data)
        img = d['image']
        lbl = d['label']

        idx = np.argwhere(lbl==2) #location of liver
        tumor_size = idx.shape[0]

        # proba = min_tumor_size / (tumor_size + 1) 
        # if proba > 1: proba = 1
        # proba = 2*(proba - 0.5)
        # if proba < 0: proba = 0
        proba = 1

        #rs = np.random.random_sample()
        if proba >= 0:
            img, lbl = add_tumor_to_slice(img, lbl, self.t_type, self.model, self.latent_size, self.tumor_shape)#, self.device)

        d['image'] = img
        d['label'] = lbl
        return d