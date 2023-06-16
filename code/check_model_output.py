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
            
            #get predicted output
            test_outputs = model(test_inputs)
            test_outputs = predict_segmentation(test_outputs, mutually_exclusive=True) # .detach().cpu()

            one_hot_out = one_hot(test_outputs, num_classes=3, dim=1)

            #find overall dice score between truth and predicted
            one_hot_labels = one_hot(test_labels, num_classes=3, dim=1)
            dice_metric(y_pred=one_hot_out, y=one_hot_labels)

            x = dice_metric.get_buffer() #just has the dice values so we print it in plot

            #split into seperate tumor instances
            for b_item in range(test_labels.shape[0]):
                tumors = test_labels[b_item,0,:,:].cpu()
                tumors[tumors == 1] = 0
                gt_seperated_tumor_labels, gt_num_regions = seperate_instances(label_image = tumors, background=0, return_num=True, connectivity=None)
                tumors = test_outputs[b_item,0,:,:].cpu()
                tumors[tumors == 1] = 0
                pred_seperated_tumor_labels, pred_num_regions = seperate_instances(label_image = tumors, background=0, return_num=True, connectivity=None)

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

            
            #plot slices
            
            if i == 0: #should just be 1 batch. batch size = number of test data
                for j in range(num_test_files):
                    txt = "Liver dice: " + str(round(x[j][0].item(),3)) + ", Tumor dice: "+ str(round(x[j][1].item(),3))
                    plt.figure("Comparison", (18, 6))
                    plt.axis('off')
                    plt.suptitle(txt)       
                    plt.subplot(1, 3, 1)
                    plt.title(f"image")
                    plt.imshow(test_inputs[j,0,:,:].detach().cpu(), cmap="gray")

                    plt.subplot(1, 3, 2)
                    plt.title(f"label")
                    plt.imshow(test_labels[j,0,:,:].detach().cpu())

                    plt.subplot(1, 3, 3)
                    plt.title(f"prediction")
                    plt.imshow(test_outputs[j,0,:,:].detach().cpu())

                    fpath = test_data['image_meta_dict']['filename_or_obj'][j]
                    fpath = fpath[-7:-4]
                    if fpath[0] == "e": fpath = fpath[2:]
                    if fpath[0] == "-": fpath = fpath[1:]
                    fname = "test-comparisons/pred-" + fpath + ".png"
                    plt.savefig(os.path.join(save_path, fname), bbox_inches='tight')
                    plt.close()
        
        ##
        with open(os.path.join(save_path,'tumor-dice-v-avg-size.pkl'),'wb') as f:
            pickle.dump(tumors_detected, f)

        plt.figure("dice v avg size")
        plt.suptitle("test set tumor dice compared to avg tumor size")
        plt.xlabel('size (pixels)')
        plt.ylabel('avg_dice')
        plt.scatter(avg_size_v_tumor_dice[:,0], avg_size_v_tumor_dice[:,1])
        plt.savefig(os.path.join(save_path, "tumor-dice-v-avg-size"), bbox_inches='tight')
        plt.close()

        ## todo: make the graph better # cummulative freq?
        # todo: if use bar then set not found -1
        with open(os.path.join(save_path,'detected-v-tumor-size.pkl'),'wb') as f:
            pickle.dump(tumors_detected, f)

        print(tumors_detected)
        plt.figure("detected v tumor_size")
        plt.suptitle("test set detected vs avg tumor size")
        plt.xlabel('size (pixels)')
        plt.ylabel('Detected = 1')
        plt.bar(tumors_detected[:,0],tumors_detected[:,1])
        plt.savefig(os.path.join(save_path, "detected-v-tumor-size"), bbox_inches='tight')
        plt.close()

        #cumulative
        # arrange tumors_detected smallest to largest
        # x sizes
        # y, sum += for 1's


        metric = dice_metric.aggregate(reduction="mean_batch") #gets the avg for liver and tumor seperately
        dice_metric.reset()
        print(f"Test set liver mean dice: {metric[0].item():.4f}")
        print(f"Test set tumor mean dice: {metric[1].item():.4f}")

    return [metric[0].item(), metric[1].item()]
