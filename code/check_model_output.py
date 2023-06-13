from imports import *

def check_model_output(save_path, model, dice_metric, data_loader, device, num_test_files):
    model_path = os.path.join(save_path, "best_metric_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
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

            #todo:dice func
            #todo save info

            #split into seperate tumor instances
            for b_item in len(test_labels[0]):
                tumors = test_labels[b_item,0,:,:]
                tumors[tumors == 1] = 0
                gt_seperated_tumor_labels, gt_num_regions = seperate_instances(label_image = tumors, background=0, return_num=True, connectivity=None)
                tumors = test_outputs[n,0,:,:]
                tumors[tumors == 1] = 0
                pred_seperated_tumor_labels, pred_num_regions = seperate_instances(label_image = tumors, background=0, return_num=True, connectivity=None)

                #for each tumor find the dice score against each tumor lbl in prediction
                tumors_detected = []
                for gt_num in gt_num_regions:   
                    individual_tumor_dice = []
                    for pred_num in pred_num_regions:
                        gt = np.where(gt_seperated_tumor_labels == gt_num+1, gt_seperated_tumor_labels, 0)
                        pred = np.where(pred_seperated_tumor_labels == pred_num+1, pred_seperated_tumor_labels, 0)
                        individual_tumor_dice.append(calc_dice(gt = gt, pred = pred))
                    
                    print(individual_tumor_dice)
                    if 1 in np.where(0.5 < individual_tumor_dice, 1, 0):
                        tumors_detected.append(1)
                    else:
                        tumors_detected.append(0)

            # save a structure containing tumor sizes and if they were detected or not



            #plot slices
            x = dice_metric.get_buffer() #just has the dice values so we print it in plot
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
                    if fpath[0] == "-": fpath = fpath[1:]
                    fname = "test-comparisons/pred-" + fpath + ".png"
                    plt.savefig(os.path.join(save_path, fname), bbox_inches='tight')
                    plt.close()

        metric = dice_metric.aggregate(reduction="mean_batch") #gets the avg for liver and tumor seperately
        dice_metric.reset()
        print(f"Test set liver mean dice: {metric[0].item():.4f}")
        print(f"Test set tumor mean dice: {metric[1].item():.4f}")
    return [metric[0].item(), metric[1].item()]
