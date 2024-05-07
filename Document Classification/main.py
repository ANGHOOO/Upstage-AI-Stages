# trainer.py
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch

class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, model_path, model_name, patience, run, scheduler, mixup_fn=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.model_name = model_name
        self.patience = patience
        self.run = run
        self.scheduler = scheduler
        self.mixup_fn = mixup_fn

    def training(self, epoch):
        self.model.train()
        train_loss = 0.0
        preds_list = []
        targets_list = []

        tbar = tqdm(self.train_dataloader)
        for batch in tbar:
            images, labels = batch

            if images is None or labels is None:
                continue

            images = images.type(torch.cuda.FloatTensor)
            images, labels = images.to(self.device), labels.to(self.device)

            if self.mixup_fn is not None:
                images, labels = self.mixup_fn(images, labels)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            if self.mixup_fn is None:
                preds_list.extend(outputs.argmax(dim=1).detach().cpu().numpy())
                targets_list.extend(labels.detach().cpu().numpy())

            tbar.set_description(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss : {loss.item():.4f}")

        train_loss = train_loss / len(self.train_dataloader)
        train_acc = None
        train_f1 = None

        if self.mixup_fn is None:
            train_acc = accuracy_score(targets_list, preds_list) if len(targets_list) > 0 else 0
            train_f1 = f1_score(targets_list, preds_list, average='macro') if len(targets_list) > 0 else 0

        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1
        }

        return metrics

    def evaluation(self, epoch):
        self.model.eval()
        valid_loss = 0.0
        preds_list = []
        targets_list = []
        batch_count = 0

        with torch.no_grad():
            tbar = tqdm(self.valid_dataloader)
            for batch in tbar:
                images, labels = batch

                if images is None or labels is None:
                    continue

                images = images.type(torch.cuda.FloatTensor)
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                valid_loss += loss.item()
                batch_count += 1

                preds_list.extend(outputs.argmax(dim=1).detach().cpu().numpy())
                targets_list.extend(labels.detach().cpu().numpy())

                tbar.set_description(f"Epoch [{epoch+1}/{self.num_epochs}] Valid Loss : {valid_loss/batch_count:.4f}")

        if batch_count > 0:
            valid_loss /= batch_count
            valid_acc = accuracy_score(preds_list, targets_list)
            valid_f1 = f1_score(preds_list, targets_list, average='macro')
        else:
            valid_loss = None
            valid_acc = None
            valid_f1 = None

        metrics = {
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
            'valid_f1': valid_f1
        }

        return metrics

    def training_loop(self):
        best_valid_loss = float('inf')
        valid_max_accuracy = -1
        valid_max_f1 = -1
        early_stop_counter = 0

        for epoch in range(self.num_epochs):
            train_metrics = self.training(epoch)
            valid_metrics = self.evaluation(epoch)
            self.scheduler.step()

            monitoring_value = {
                'train_loss': train_metrics['train_loss'],
                'valid_loss': valid_metrics['valid_loss']
            }
            self.run.log(monitoring_value, step=epoch)

            if valid_metrics['valid_loss'] < best_valid_loss:
                best_valid_loss = valid_metrics['valid_loss']
                early_stop_counter = 0
                torch.save(self.model.state_dict(), f"{self.model_path}/model_{self.model_name}.pt")
                self.run.summary['best_train_loss'] = train_metrics['train_loss']
                self.run.summary['best_valid_loss'] = valid_metrics['valid_loss']
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.patience:
                print('Early Stopping!')
                break

        return valid_max_accuracy, valid_max_f1