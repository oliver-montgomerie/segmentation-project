from imports import *

# def Hausdorff(sitk_pred, sitk_gt, show_print=True):
#     hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
#     hausdorffcomputer.Execute(sitk_pred, sitk_gt)
#     hd = hausdorffcomputer.GetHausdorffDistance()
#     if show_print:
#         print('Hausdorff distance:', np.around(hd,2))
#     ahd = hausdorffcomputer.GetAverageHausdorffDistance()
#     if show_print:
#         print('Average Hausdorff distance:', np.around(ahd,2))
#     return (hd, ahd)

def calc_dice(pred, gt, show_print=False):
    intersection=np.sum(pred*gt)
    union = np.sum(pred)+np.sum(gt)
    d = 2*intersection/union
    if show_print:
        print('Dice =',np.round(d,2))
    return d


def check_model_output(save_path, model, dice_metric, data_loader, device, num_test_files):
    model_path = os.path.join(save_path, "best_metric_model.pth")
    #model_path = os.path.join(save_path, "epoch-3-model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    tumors_detected = np.empty((0,2), float)
    avg_size_v_tumor_dice = np.empty((0,2), float)
    with torch.no_grad():
        for i, test_data in enumerate(data_loader):
            test_inputs, test_labels = (
                        test_data["image"].to(device),
                        test_data["label"].to(device),
                    )
            running_batch_size = test_labels.shape[0]
            
            #get predicted output
            test_outputs = model(test_inputs)
            
            # plt.figure("Comparison", (18, 6))      
            # plt.subplot(2, 3, 1)
            # plt.title(f"input")
            # plt.imshow(test_inputs[0,0,:,:].detach().cpu())

            # plt.subplot(2, 3, 2)
            # plt.title(f"output")
            # plt.imshow(test_outputs[0,0,:,:].detach().cpu())
            # plt.show()
            # plt.pause(1)

            test_outputs = predict_segmentation(test_outputs, mutually_exclusive=True) # .detach().cpu()

            one_hot_out = one_hot(test_outputs, num_classes=3, dim=1)

            # plt.subplot(2, 3, 4)
            # plt.imshow(one_hot_out[0,0,:,:].detach().cpu())
            # plt.subplot(2, 3, 5)
            # plt.imshow(one_hot_out[0,1,:,:].detach().cpu())
            # plt.subplot(2, 3, 6)
            # plt.imshow(one_hot_out[0,2,:,:].detach().cpu())

            #find overall dice score between truth and predicted
            one_hot_labels = one_hot(test_labels, num_classes=3, dim=1)

            # plt.subplot(2, 3, 1)
            # plt.imshow(one_hot_labels[0,0,:,:].detach().cpu())
            # plt.subplot(2, 3, 2)
            # plt.imshow(one_hot_labels[0,1,:,:].detach().cpu())
            # plt.subplot(2, 3, 3)
            # plt.imshow(one_hot_labels[0,2,:,:].detach().cpu())
            # plt.show()
            # plt.pause(1)

            dice_metric(y_pred=one_hot_out, y=one_hot_labels)

            x = dice_metric.get_buffer() #just has the dice values so we print it in plot
            x = x[-running_batch_size:]
            
            for b_item in range(running_batch_size):
                test_file_counter = (i * running_batch_size) + b_item

                #split into seperate tumor instances
                tumors = test_labels[b_item,0,:,:].cpu()
                tumors[tumors == 1] = 0
                gt_seperated_tumor_labels, gt_num_regions = seperate_instances(label_image = tumors, background=0, return_num=True, connectivity=None)
                tumors = test_outputs[b_item,0,:,:].cpu()
                tumors[tumors == 1] = 0
                pred_seperated_tumor_labels, pred_num_regions = seperate_instances(label_image = tumors, background=0, return_num=True, connectivity=None)

                #save  avg tumor size and dice prediction of slice
                gt_avg_tumor_size = np.sum(gt_seperated_tumor_labels > 0) / gt_num_regions
                avg_size_v_tumor_dice = np.append(avg_size_v_tumor_dice, [[gt_avg_tumor_size, x[b_item][1].item()]], axis = 0)

                #for each tumor find the dice score against each tumor lbl in prediction
                for gt_num in range(gt_num_regions):   
                    individual_tumor_dice = np.array([])
                    for pred_num in range(pred_num_regions):
                        gt = np.where(gt_seperated_tumor_labels == gt_num+1, gt_seperated_tumor_labels, 0)
                        pred = np.where(pred_seperated_tumor_labels == pred_num+1, pred_seperated_tumor_labels, 0)
                        individual_tumor_dice = np.append(individual_tumor_dice, calc_dice(gt = gt, pred = pred))
                    
                    gt_tumor_size = np.sum(gt_seperated_tumor_labels == gt_num+1)
            
                    # save a structure containing tumor sizes and if they were detected or not
                    if 1 in np.where(0.5 < individual_tumor_dice, 1, 0):
                        tumors_detected = np.append(tumors_detected, [[gt_tumor_size, 1]], axis=0)
                    else:
                        tumors_detected = np.append(tumors_detected, [[gt_tumor_size, 0]], axis=0)

                #plot 20 slices from range of files
                if test_file_counter in np.linspace(0, num_test_files, 20, dtype=int):
                    txt = "Liver dice: " + str(round(x[b_item][0].item(),3)) + ", Tumor dice: "+ str(round(x[b_item][1].item(),3)) + "\nTumor pixels: " + str(np.sum(gt_seperated_tumor_labels > 0))
                    plt.figure("Comparison", (18, 6))
                    plt.suptitle(txt)       
                    plt.subplot(1, 3, 1)
                    plt.axis('off')
                    plt.title(f"image")
                    plt.imshow(test_inputs[b_item,0,:,:].detach().cpu(), cmap="gray")

                    plt.subplot(1, 3, 2)
                    plt.axis('off')
                    plt.title(f"label")
                    plt.imshow(test_labels[b_item,0,:,:].detach().cpu(), vmin=0, vmax=2)

                    plt.subplot(1, 3, 3)
                    plt.axis('off')
                    plt.title(f"prediction")
                    plt.imshow(test_outputs[b_item,0,:,:].detach().cpu(), vmin=0, vmax=2)

                    fpath = test_data['image_meta_dict']['filename_or_obj'][b_item]
                    fpath = fpath[fpath.rfind("/")+1:-4] 
                    fname = "test-comparisons/pred-" + fpath + ".png"
                    plt.savefig(os.path.join(save_path, fname), bbox_inches='tight')
                    plt.close()
                    # plt.show()
                    # plt.pause(1)

        print(f"{i}/{len(data_loader)}")
        

    # ## dice v avg size
    # avg_size_v_tumor_dice = sorted(avg_size_v_tumor_dice, key=lambda id: id[0])
    # avg_size_v_tumor_dice = np.array(avg_size_v_tumor_dice)
    # with open(os.path.join(save_path,'tumor-dice-v-avg-size.pkl'),'wb') as f:
    #     pickle.dump(avg_size_v_tumor_dice, f)

    # plt.figure("dice v avg size")
    # plt.suptitle("test set tumor dice compared to avg tumor size")
    # plt.xlabel('size (pixels)')
    # plt.ylabel('avg_dice')
    # plt.scatter(avg_size_v_tumor_dice[:,0], avg_size_v_tumor_dice[:,1])
    # plt.savefig(os.path.join(save_path, "tumor-dice-v-avg-size"), bbox_inches='tight')
    # plt.close()

    # num_bins = 20
    # num_in_bin = len(avg_size_v_tumor_dice) // num_bins + 1
    # bin_avg_dice = np.zeros(num_bins)
    # bin_centres = np.zeros(num_bins)
    # width = np.zeros((num_bins,2))
    # width[:,0] = np.inf
    # for i, t in enumerate(avg_size_v_tumor_dice):
    #     bin_num = i // num_in_bin
    #     if bin_num >= num_bins: break
    #     bin_avg_dice[bin_num] += t[1]
    #     bin_centres[bin_num] += t[0]
        
    #     if t[0] < width[bin_num,0]:
    #         width[bin_num,0] = t[0]
    #     if t[0] > width[bin_num,1]:
    #         width[bin_num,1] = t[0]

    # widths = [(x[1] - x[0]) -20 for x in width]

    # bin_avg_dice = [x / num_in_bin for x in bin_avg_dice]
    # bin_centres = [(x[1] + x[0])/2 for x in width]

    # plt.figure("avg dice v tumor_size")
    # plt.suptitle("test set avg dice vs avg tumor size")
    # plt.xlabel('average size (pixels)')
    # plt.ylabel('avg dice')
    # plt.bar(bin_centres, bin_avg_dice, width= widths)
    # plt.savefig(os.path.join(save_path, "avg-tumor-dice-v-avg-size"), bbox_inches='tight')
    # plt.close()


    # ## detection v size
    # tumors_detected = sorted(tumors_detected, key=lambda id: id[0])
    # tumors_detected = np.array(tumors_detected)
    # with open(os.path.join(save_path,'detected-v-tumor-size.pkl'),'wb') as f:
    #     pickle.dump(tumors_detected, f)

    # y = np.zeros(len(tumors_detected))
    # y_sum = 0
    # for t in range(len(tumors_detected)):
    #     y_sum += tumors_detected[t][1]
    #     y[t] = y_sum

    # plt.figure("detected v tumor_size")
    # plt.suptitle("test set detected vs avg tumor size")
    # plt.xlabel('size (pixels)')
    # plt.ylabel('Cumulative detections')
    # plt.plot(tumors_detected[:,0], y)
    # plt.savefig(os.path.join(save_path, "detected-v-tumor-size-cumulative"), bbox_inches='tight')
    # plt.close()

    # num_bins = 20
    # num_in_bin = len(tumors_detected) // num_bins + 1
    # bin_perc_detected = np.zeros(num_bins)
    # bin_centres = np.zeros(num_bins)
    # width = np.zeros((num_bins,2))
    # width[:,0] = np.inf
    # for i, t in enumerate(tumors_detected):
    #     bin_num = i // num_in_bin
    #     if bin_num >= num_bins: break
    #     bin_perc_detected[bin_num] += t[1]
    #     bin_centres[bin_num] += t[0]
        
    #     if t[0] < width[bin_num,0]:
    #         width[bin_num,0] = t[0]
    #     if t[0] > width[bin_num,1]:
    #         width[bin_num,1] = t[0]

    # widths = [(x[1] - x[0]) -20 for x in width]

    # bin_perc_detected = [100 * x / num_in_bin for x in bin_perc_detected]
    # bin_centres = [(x[1] + x[0])/2 for x in width]

    # plt.figure("detected v tumor_size")
    # plt.suptitle("test set detected vs avg tumor size")
    # plt.xlabel('average size (pixels)')
    # plt.ylabel('percentage detected')
    # plt.bar(bin_centres, bin_perc_detected, width=widths)
    # plt.savefig(os.path.join(save_path, "detected-v-tumor-size-bars"), bbox_inches='tight')
    # plt.close()   


    ## print n return metrics.
    metric = dice_metric.aggregate(reduction="mean_batch") #gets the avg for liver and tumor seperately
    dice_metric.reset()
    print(f"Test set liver mean dice: {metric[0].item():.4f}")
    print(f"Test set tumor mean dice: {metric[1].item():.4f}")

    return [metric[0].item(), metric[1].item()]



# # #save_path="/home/omo23/Documents/segmentation-project/saved-tests/test"
# save_path = "/home/omo23/Documents/segmentation-project/saved-tests/0-100-normal-REAL"
# num_workers = 4
# batch_size = 16 

# #Data loading
# data_dir = "/home/omo23/Documents/sliced-data"
# all_images = sorted(glob.glob(os.path.join(data_dir, "Images", "*.nii")))
# all_labels = sorted(glob.glob(os.path.join(data_dir, "Labels", "*.nii")))
# data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)]

# # filter out slices with small tumor area
# data_dicts = [item for item in data_dicts if file_tumor_size(item) > min_tumor_size]

# test_files = []
# for d in data_dicts:
#     d_num = d['image']
#     d_num = d_num[d_num.rfind("/")+1:d_num.rfind("-")] 
#     if d_num in test_files_nums:
#         test_files.append(d)

# ## sort test files smallest to largest tumor
# test_files = sorted(test_files, key=lambda file: file_tumor_size(file))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = UNet(
#     spatial_dims=2,
#     in_channels=1,
#     out_channels=3,
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
#     norm=Norm.BATCH,
# ).to(device)

# from transforms import test_transforms
# #test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=num_workers)
# test_ds = Dataset(data=test_files, transform=test_transforms)
# test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# testset_dice_metric = DiceMetric(include_background=False, reduction="none")

# testset_dice = check_model_output(save_path = save_path, 
#                     model = model, 
#                     dice_metric = testset_dice_metric,
#                     data_loader = test_loader,
#                     device = device,
#                     num_test_files = len(test_files))