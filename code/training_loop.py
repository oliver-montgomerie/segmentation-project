from imports import *

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print(f"Early stop counter: {self.counter}/{self.patience}")
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print(f"Early stop counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
        return False
    

def save_train_images(batch_data, save_path):
    images, _ = (
        batch_data["image"],
        batch_data["label"],
    )
    for i in range(images.shape[0]):
        plt.figure("Training data", (18, 6))
        plt.axis('off')
        plt.imshow(images[i,0,:,:].detach().cpu(), cmap="gray")
        fpath = batch_data['image_meta_dict']['filename_or_obj'][i]
        fpath = fpath[fpath.rfind("/")+1:-4] 
        fname = "training-batch/img-" + fpath + ".png"
        plt.savefig(os.path.join(save_path, fname), bbox_inches='tight')
        plt.close()

    return


def training_loop(model,
                  train_loader,
                  val_loader,
                  optimizer,
                  scheduler,
                  loss_function,
                  dice_metric,
                  device,
                  save_path,
                  max_epochs = 1,
                  ):

    early_stopper = EarlyStopper(patience=10, min_delta=0)
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

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
            if step == 1 and epoch == 0:    #save first batch of training data
                save_train_images(batch_data, save_path)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_loader)}, " f"train_loss: {loss.item():.4f}")

        scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        #validation 
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = model(val_inputs)
                    val_outputs = predict_segmentation(val_outputs, mutually_exclusive=True)
                    val_outputs = one_hot(val_outputs, num_classes=3, dim=1)

                    val_labels = one_hot(val_labels, num_classes=3, dim=1)
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(save_path, "best_metric_model.pth")) 
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current val mean dice: {metric:.4f}"
                    f"\nbest val mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

                ## early stopping
                if early_stopper.early_stop(1-metric): 
                    print("### stopping early ###")            
                    break

    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    return epoch_loss_values, metric_values