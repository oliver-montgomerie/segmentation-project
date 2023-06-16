from imports import *

def stuff():
    img_save_path = "/home/omo23/Documents/sliced-data/Images"
    lbl_save_path = "/home/omo23/Documents/sliced-data/Labels"

    for i in range(131):
        img_path = "/data/datasets/Liver/LiTS2017/Volumes/volume-" + str(i) + ".nii"
        lbl_path = "/data/datasets/Liver/LiTS2017/Segmentations/segmentation-" + str(i) + ".nii"
        img = nib.load(img_path)
        lbl = nib.load(lbl_path)

        np_img = np.array(img.dataobj)
        np_lbl = np.array(lbl.dataobj)

        tumor_in_slice = np.sum((np_lbl == 2).reshape(-1, np_lbl.shape[-1]), axis=0)
        #print(tumor_in_slice)
        for idx, tumor_pixels in enumerate(tumor_in_slice):
            if tumor_pixels > 0:
                print(f"Saving img: {i:3.0f} slice: {idx:3.0f}, with {tumor_pixels} tumor pixels")
                ni_img = nib.Nifti1Image(np_img[:,:,idx], img.affine) #, img.header (?)
                ni_lbl = nib.Nifti1Image(np_lbl[:,:,idx], lbl.affine)
                nib.save(ni_img, os.path.join(img_save_path, str(i) + "-" + str(idx) + ".nii"))
                nib.save(ni_lbl, os.path.join(lbl_save_path, str(i) + "-" + str(idx) + ".nii"))

if __name__ == '__main__':
    stuff()