import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import torchvision
from components.CLIPBert import  ClipBertForSequenceClassification
from components.Resnet import GridFeatBackbone

class ClipBert(pl.LightningModule):
    def __init__(self, config, learning_rate, num_classes):
        super().__init__()
        self.config = config
        self.cnn = GridFeatBackbone()
        self.transformer = ClipBertForSequenceClassification(self.config)

        self.lr = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, batch):
        visual_features = self.cnn(batch["visual_inputs"])
        batch["visual_inputs"] = visual_features
        output = self.transformer(**batch)  # dict
        return output

    
    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False


    def training_step(self, batch, batch_idx):
        
        loss, logits, y = self._common_step(batch, batch_idx)
        
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        return {"loss": loss, "scores": logits, "y": y}
    
    def training_epoch_end(self, outputs):

        scores = torch.cat([x["scores"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        self.log_dict(
            {
                "train_acc": self.accuracy(scores, y),
                "train_f1": self.f1_score(scores, y),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        
        loss, logits, y = self._common_step(batch, batch_idx)
        val_acc = self.accuracy(logits, y)
        val_f1 = self.f1_score(logits, y)
        self.log("val_loss", loss)
        self.log("val_acc", val_acc)
        self.log("val_f1", val_f1)
        
        return loss

    def test_step(self, batch, batch_idx):
        
        loss, logits, y = self._common_step(batch, batch_idx)
        test_acc = self.accuracy(logits, y)
        test_f1 = self.f1_score(logits, y)
        self.log("test_loss", loss)
        self.log("test_acc", test_acc)
        self.log("test_f1", test_f1)
        
        return loss

    def _common_step(self, batch, batch_idx):
        
        text = batch['text']
        ids = text['input_ids'].squeeze(1)
        masks = text['attention_mask'].squeeze(1)
        y = batch['label']
        inp_batch = {
            'text_input_ids': ids,
            'visual_inputs': batch['img'],
            'text_input_mask': masks,
            'labels': y
        }
        logits = self.forward(inp_batch)
        loss = self.loss_fn(
                        logits.view(-1, self.config.num_labels),
                        y.view(-1))
        
        return loss, logits, y

    def predict_step(self, batch, batch_idx):
        _, logits, _ = self._common_step(batch, batch_idx)
        return logits

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)