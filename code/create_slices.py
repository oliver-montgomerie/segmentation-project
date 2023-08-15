from imports import *

min_liver_size = 707 # pi  225 * PI for 30mm diameter  15mm rad

def stuff():
    # img_save_path = "C:/Users/olive/OneDrive/Desktop/Liver Files/sliced-data/Images"
    # lbl_save_path = "C:/Users/olive/OneDrive/Desktop/Liver Files/sliced-data/Labels"
    img_save_path = "/home/omo23/Documents/sliced-data/Images"
    lbl_save_path = "/home/omo23/Documents/sliced-data/Labels"
    
    for i in range(131):
        print(i)
        # i=48
        # img_path = "C:/Users/olive/OneDrive/Desktop/Liver Files/imagesTr/volume-" + str(i) + ".nii"  
        # lbl_path = "C:/Users/olive/OneDrive/Desktop/Liver Files/labelsTr/segmentation-" + str(i) + ".nii"
        img_path = "/data/datasets/Liver/LiTS2017/Volumes/volume-" + str(i) + ".nii"
        lbl_path = "/data/datasets/Liver/LiTS2017/Segmentations/segmentation-" + str(i) + ".nii"
        img = nib.load(img_path)
        lbl = nib.load(lbl_path)

        # print(img.header)
        # print(lbl.header)

        # print(img.affine)
        # print(lbl.affine)

        np_img = np.array(img.dataobj)
        np_lbl = np.array(lbl.dataobj)

        for z in range(np_lbl.shape[-1]):
            lbl_slice = np_lbl[:,:,z]
            liver_size = np.sum(lbl_slice == 1) * img.header['pixdim'][1] * img.header['pixdim'][2]
            if liver_size > min_liver_size:
                ni_img = nib.Nifti1Image(np_img[:,:,z], img.affine, img.header) #, img.header (?)
                ni_lbl = nib.Nifti1Image(np_lbl[:,:,z], img.affine, img.header)
                nib.save(ni_img, os.path.join(img_save_path, str(i) + "-" + str(z) + ".nii"))
                nib.save(ni_lbl, os.path.join(lbl_save_path, str(i) + "-" + str(z) + ".nii"))


        # tumor_in_slice = np.sum((np_lbl == 2).reshape(-1, np_lbl.shape[-1]), axis=0)
        # #print(tumor_in_slice)
        # for idx, tumor_pixels in enumerate(tumor_in_slice):
        #     if tumor_pixels > 0:
        #         print(f"Saving img: {i:3.0f} slice: {idx:3.0f}, with {tumor_pixels} tumor pixels")
        #         ni_img = nib.Nifti1Image(np_img[:,:,idx], img.affine) #, img.header (?)
        #         ni_lbl = nib.Nifti1Image(np_lbl[:,:,idx], lbl.affine)
        #         nib.save(ni_img, os.path.join(img_save_path, str(i) + "-" + str(idx) + ".nii"))
        #         nib.save(ni_lbl, os.path.join(lbl_save_path, str(i) + "-" + str(idx) + ".nii"))

if __name__ == '__main__':
    stuff()