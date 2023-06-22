from imports import *
import matplotlib.pyplot as plt

file = "/home/omo23/Documents/segmentation-project/saved-tests/60-20-20-no-transform/detected-v-tumor-size.pkl"

with open(file, "rb") as input_file:
    tumors_detected = pickle.load(input_file)

tumors_detected = sorted(tumors_detected, key=lambda id: id[0])
tumors_detected = np.array(tumors_detected)

#tumors detected in order
y = np.zeros(len(tumors_detected))
y_sum = 0
for t in range(len(tumors_detected)):
    y_sum += tumors_detected[t][1]
    y[t] = y_sum

plt.figure("detected v tumor_size")
plt.suptitle("test set detected vs avg tumor size")
plt.xlabel('size (pixels)')
plt.ylabel('Cummulative detections')
plt.plot(tumors_detected[:,0], y)
plt.show()

# plt.figure("detected v tumor_size")
# plt.suptitle("test set detected vs avg tumor size")
# plt.xlabel('size (pixels)')
# plt.ylabel('Detected = 1')
# plt.scatter(tumors_detected[:,0],tumors_detected[:,1])
# plt.show()

num_bins = 20
num_in_bin = len(tumors_detected) // num_bins + 1
bin_perc_detected = np.zeros(num_bins)
bin_centres = np.zeros(num_bins)
width = np.zeros((num_bins,2))
width[:,0] = np.inf
for i, t in enumerate(tumors_detected):
    bin_num = i // num_in_bin
    if bin_num >= num_bins: break
    bin_perc_detected[bin_num] += t[1]
    bin_centres[bin_num] += t[0]
    
    if t[0] < width[bin_num,0]:
        width[bin_num,0] = t[0]
    if t[0] > width[bin_num,1]:
        width[bin_num,1] = t[0]

widths = [(x[1] - x[0]) -20 for x in width]

bin_perc_detected = [100 * x / num_in_bin for x in bin_perc_detected]
#bin_centres = [x // num_in_bin for x in bin_centres]
bin_centres = [(x[1] + x[0])/2 for x in width]

plt.figure("detected v tumor_size")
plt.suptitle("test set detected vs avg tumor size")
plt.xlabel('average size (pixels)')
plt.ylabel('percentage detected')
plt.bar(bin_centres, bin_perc_detected, width=widths)
plt.show()
#plt.savefig(os.path.join(save_path, "detected-v-tumor-size-bars"), bbox_inches='tight')
#plt.close()



file = "/home/omo23/Documents/segmentation-project/saved-tests/60-20-20-no-transform/tumor-dice-v-avg-size.pkl"

with open(file, "rb") as input_file:
    avg_size_v_tumor_dice = pickle.load(input_file)

avg_size_v_tumor_dice = sorted(avg_size_v_tumor_dice, key=lambda id: id[0])
avg_size_v_tumor_dice = np.array(avg_size_v_tumor_dice)

num_bins = 20
num_in_bin = len(avg_size_v_tumor_dice) // num_bins + 1
bin_avg_dice = np.zeros(num_bins)
bin_centres = np.zeros(num_bins)
width = np.zeros((num_bins,2))
width[:,0] = np.inf
for i, t in enumerate(avg_size_v_tumor_dice):
    bin_num = i // num_in_bin
    if bin_num >= num_bins: break
    bin_avg_dice[bin_num] += t[1]
    bin_centres[bin_num] += t[0]
    
    if t[0] < width[bin_num,0]:
        width[bin_num,0] = t[0]
    if t[0] > width[bin_num,1]:
        width[bin_num,1] = t[0]

widths = [(x[1] - x[0]) -20 for x in width]

bin_avg_dice = [x / num_in_bin for x in bin_avg_dice]
#bin_centres = [x // num_in_bin for x in bin_centres]
bin_centres = [(x[1] + x[0])/2 for x in width]

plt.figure("avg dice v tumor_size")
plt.suptitle("test set avg dice vs avg tumor size")
plt.xlabel('average size (pixels)')
plt.ylabel('avg dice')
plt.bar(bin_centres, bin_avg_dice, width= widths)
plt.show()