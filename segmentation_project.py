# print("Hello")

# # a = 1
# b = 2
# c = 3

# with open('info.txt','w') as f:
#         txt = ["Test set mean dice:", a,
#                 "\nNumber of train files:", b,
#                 "\nNumber of val files:", c,
#         ]
#         for t in txt:
#             f.write(f"{t}")

# metric = [1,2,3,4,5]

# nm = list(map(lambda x:1-x, metric))

# print(nm)

# num = 1.2345
# x = round(num,2)

# print(x)
# import matplotlib.pyplot as plt

# def testfunc(a):
#     plt.figure("test", (12, 6))
#     plt.plot([1,2,3,4,5], [1+a,2+a,3+a,4+a,5+a], label="train")
#     plt.show()


# testfunc(0)
# testfunc(3)

# x = 224.34
# print(round(x))

# s= 'hi'
# print("%7s" % s)
# s = "%7s" % s
# print(s)
# fpath = "a"

# print(f"FILE: {fpath:3s} ok")

# x = 123.456
# y = 12.2
# print(f"{x:3.0f}a")
# print(f"{y:3.0f}a")

# ##write pixdimensinos
# import nibabel as nib
# for i in range(131):
#     path = "/data/datasets/Liver/LiTS2017/Volumes/volume-"
#     path += str(i) + ".nii"
#     img = nib.load(path)
#     #print(img.header['dim'])
#     print(f"{i:3.0f}: {img.header['pixdim'][1]:.3f}, {img.header['pixdim'][2]:.3f}, {img.header['pixdim'][3]:.3f}")

# if 1 in np.where(0.5 < individual_tumor_dice, 1, 0):
#                         tumors_detected.append(1)
#                     else:
#                         tumors_detected.append(0)
import numpy as np
x = np.array([0,0.1,0.4,0.4])
if 1 in np.where(0.5 < x, 1, 0):
    print(1)
else:
    print(0)