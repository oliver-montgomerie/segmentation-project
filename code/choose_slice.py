from imports import *
from monai.transforms import MapTransform

#Chooses the slice with the largest amount of label
class SliceWithMaxNumLabelsd(MapTransform):
    def __init__(self, keys, label_key):
        self.keys = keys
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)
        im = d[self.label_key]
        q = np.sum((im == 2).reshape(-1, im.shape[-1]), axis=0)
        _slice = np.where(q == np.max(q))[0][0]
        for key in self.keys:
            d[key] = d[key][..., _slice]
        return d


# Save slice
class SaveSliced(MapTransform):
    def __init__(self, keys, path):
        self.keys = keys
        self.path = path

    def __call__(self, data):
        d = {}
        for key in self.keys:
            fname = os.path.basename(
                data[key + "_meta_dict"]["filename_or_obj"])
            path = os.path.join(self.path, key, fname)
            nib.save(nib.Nifti1Image(data[key], np.eye(4)), path)
            d[key] = path
        return d