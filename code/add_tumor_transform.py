from imports import *

def generate_a_tumor(model, latent_size, dist, tumor_shape): #, device):
    with torch.no_grad():
        sample = torch.zeros(1, latent_size) #.to(device)
        for s in range(sample.shape[1]):
            #maybe This should be weighted towards the edges for unusual looking tumors?
            sample[0,s] = dist.icdf((torch.rand(1)*0.9) + 0.05)

        o = model.decode_forward(sample) #get output from latent sample
        o = o.detach().cpu().numpy().reshape(tumor_shape)
        mask = np.zeros(o.shape)    #create mask from thresholding tumor
        thresh = (np.max(o) + np.min(o))/2
        mask[o > thresh] = 1
        #clip the tumor to the mask size
        o[mask != 1] = 0

        # # make sure its large enough
        # tumor_size = np.sum(mask == 1) * 0.793 * 0.793
        # tumor_size = int(tumor_size)
        # if tumor_size < min_tumor_size:
        #     continue

        return torch.from_numpy(o), torch.from_numpy(mask) #o, mask 
    
def get_real_tumor(): #, device):
    #check through tumor pat
    folder = "/home/omo23/Documents/tumor-patches-data/Images/"
    got_tumor = False
    
    while not got_tumor:
        file = random.choice(os.listdir(folder))
        fnum = file[:file.rfind("-")] 
        if fnum in train_files_nums:
            got_tumor = True
            tumor = nib.load(os.path.join(folder, file))
            np_tumor = np.array(tumor.dataobj)
            np_mask = np.zeros(np_tumor.shape)
            np_mask[np_tumor > np.min(np_tumor)] = 1
            
    return torch.from_numpy(np_tumor), torch.from_numpy(np_mask)


def add_tumor_to_slice(img, lbl, t_type, model, latent_size, tumor_shape): #, device):
    #todo: change function to torch instead of numpy ?why
    dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))

    liver_pix = torch.argwhere(lbl == 1)

    max_attempts = 20
    good_placement = False
    for attempt in range(max_attempts):
        if t_type == "real":
            tumor_img, tumor_lbl = get_real_tumor() #could pass train file nums?
        else:
            tumor_img, tumor_lbl = generate_a_tumor(model, latent_size, dist, tumor_shape) #, device)

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

        # Todo: add a probability for repeating the process for adding multiple tumors.
        # look at ratio in other slices and use that??

    if good_placement == False:
        return img, lbl #torch.unsqueeze(torch.from_numpy(img), 0), torch.unsqueeze(torch.from_numpy(lbl), 0)
        
    ## merge images
    sobel_h = ndimage.sobel(tumor_lbl, 0)  # horizontal gradient
    sobel_v = ndimage.sobel(tumor_lbl, 1)  # vertical gradient
    tumor_edges = np.sqrt(sobel_h**2 + sobel_v**2)
    tumor_edges = tumor_edges / np.max(tumor_edges)
    #tumor_edges = ndimage.gaussian_filter(tumor_edges, sigma = 0.25)

    edge_locations = np.argwhere(tumor_edges > 0.5)
    lbl_locations = np.argwhere(tumor_lbl >= 3)
    dists = cdist(lbl_locations, edge_locations).min(axis=1)
    distmap = np.zeros(tumor_lbl.shape)
    distmap[tumor_lbl >= 3] = dists
    distmap = distmap / (2*np.max(distmap))
    distmap[distmap>0] = distmap[distmap>0] + 0.5

    gen_img = img.detach().clone() 
    #print(gen_img.size(), tumor_lbl.shape)
    #gen_img[tumor_lbl >= 3] = tumor_img[tumor_lbl >= 3] #(0.8*tumor_img[tumor_lbl >= 3]) + (0.2*gen_img[tumor_lbl >= 3])
    gen_img[tumor_lbl >= 3] = ((distmap*tumor_img)[tumor_lbl >= 3]) + (((1-distmap)*img)[tumor_lbl >= 3])

    # plt.figure("New tumor", (18, 6))
    # plt.subplot(2,3,1)
    # plt.imshow(img.detach().cpu().numpy()[0,:,:], cmap="gray")
    # plt.title("Original")
    # plt.axis('off')
    # plt.subplot(2,3,4)
    # plt.imshow(lbl.detach().cpu().numpy()[0,:,:], vmin=0, vmax=5)
    # plt.axis('off')

    # plt.subplot(2,3,2)
    # plt.imshow(tumor_img.detach().cpu().numpy()[0,:,:], cmap="gray")
    # plt.axis('off')
    # plt.title("Generated tumor")
    # plt.subplot(2,3,5)
    # plt.imshow(tumor_lbl.detach().cpu().numpy()[0,:,:], vmin=0, vmax=5)
    # plt.axis('off')

    # plt.subplot(2,3,3)
    # plt.imshow(gen_img.detach().cpu().numpy()[0,:,:], cmap="gray")
    # plt.title("Implanted tumor")
    # plt.axis('off')
    # plt.subplot(2,3,6)
    # plt.imshow(gen_lbl.detach().cpu().numpy()[0,:,:], vmin=0, vmax=5)
    # plt.axis('off')

    # plt.show()

    gen_lbl[gen_lbl >= 3] = 2 

    return gen_img, gen_lbl



class implant_tumor(MapTransform):
    def __init__(self, keys, t_type, load_path):
        #model shape should definetly be saved in a config somewhere instead of this...
        self.keys = keys

        self.t_type = t_type #real, vae, vae-gan, diffusion...

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

        rs = np.random.random_sample()
        if proba >= rs:
            img, lbl = add_tumor_to_slice(img, lbl, self.t_type, self.model, self.latent_size, self.tumor_shape)#, self.device)
            
        d['image'] = img
        d['label'] = lbl
        return d