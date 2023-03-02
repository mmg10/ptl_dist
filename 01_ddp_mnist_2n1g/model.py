
import torch

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn

from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output
    
    
class LitResnet(pl.LightningModule):
    def __init__(self, lr, opt, num_classes):
        super().__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = Net()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        train_loss = self.loss(logits, y)
        return {"loss": train_loss, "preds": preds, "targ": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        val_loss = self.loss(logits, y)
        return {"loss": val_loss, "preds": preds, "targ": y}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        test_loss = self.loss(logits, y)
        return {"loss": test_loss, "preds": preds, "targ": y}
    
    def validation_epoch_end(self, outputs):
        loss, preds, targ = self.gatherer(outputs)
        acc = (torch.sum(torch.eq(preds,targ)) / len(preds)).item()*100
        print(f'Val Accuracy: {acc}')
        # for Tensorboard
        self.logger.experiment.add_scalar("loss/val",
                                            loss,
                                            self.current_epoch)
        self.logger.experiment.add_scalar("acc/val",
                                            acc,
                                            self.current_epoch)
        # can be used for monitoring
        self.log('val_acc', acc, sync_dist=True, logger=False)
        self.log('val_loss', loss, sync_dist=True, logger=False)
        
    def test_epoch_end(self, outputs):
        loss, preds, targ = self.gatherer(outputs)
        acc = (torch.sum(torch.eq(preds,targ)) / len(preds)).item()*100
        # for Tensorboard
        self.logger.experiment.add_scalar("loss/test",
                                            loss,
                                            self.current_epoch)
        self.logger.experiment.add_scalar("acc/test",
                                            acc,
                                            self.current_epoch)
        # can be used for monitoring
        self.log('test_acc', acc, sync_dist=True, logger=False)
        self.log('test_loss', loss, sync_dist=True, logger=False)        

        
    def training_epoch_end(self, outputs):
        
        # this line is used to get data from all GPUs
        loss, preds, targ = self.gatherer(outputs)
        
        acc = (torch.sum(torch.eq(preds,targ)) / len(preds)).item()*100
        # for Tensorboard
        self.logger.experiment.add_scalar("loss/train",
                                            loss,
                                            self.current_epoch)
        self.logger.experiment.add_scalar("acc/train",
                                            acc,
                                            self.current_epoch)
        # can be used for monitoring
        self.log('train_acc', acc, sync_dist=True, logger=False)
        self.log('train_loss', loss, sync_dist=True, logger=False)

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        
        return {"optimizer": optimizer}
    
    def gatherer(self, outputs):
        all_out = self.all_gather(outputs)
        loss = torch.cat([x['loss'].flatten().cpu() for x in all_out]).mean()
        preds = torch.cat([x['preds'].flatten().cpu() for x in all_out])
        targ = torch.cat([x['targ'].flatten().cpu() for x in all_out]) 
        return loss, preds, targ