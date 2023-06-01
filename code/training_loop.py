from imports import *
#todo: change the 5 with number of data
def training_loop(model,
                  train_loader,
                  val_loader,
                  optimizer,
                  loss_function,
                  dice_metric,
                  device,
                  max_epochs = 1,
                  ):
    root_dir = "C:/Users/olive/OneDrive/Desktop/Liver Files/segmentation-project/saved_models"
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #todo:here
            print(f"{step}/{5 // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = model(val_inputs)
                    #roi_size = (160, 160, 160)
                    #sw_batch_size = 4
                    #val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    #val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    #val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                    #todo: labels to onehot? or output argmax
                    # compute metric for current iteration
                    val_labels = one_hot(val_labels, num_classes=3, dim=0)
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    return epoch_loss_values, metric_values