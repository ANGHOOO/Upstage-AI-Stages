import torch
import tqdm
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, valid_loader, device, criterion, epoch, num_epochs):
    model.eval()
    valid_loss = 0.0
    preds_list = []
    targets_list = []
    with torch.no_grad():
        tbar = tqdm.tqdm(valid_loader)
        for batch in tbar:
            images, labels = batch
            if images is None or labels is None:  # skip corrupted images
                continue
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            preds_list.extend(outputs.argmax(dim=1).cpu().numpy())
            targets_list.extend(labels.cpu().numpy())
            tbar.set_description(f"Valid Loss : {valid_loss:.4f}")
    valid_loss = valid_loss / len(valid_loader)
    valid_acc = accuracy_score(preds_list, targets_list)
    valid_f1 = f1_score(preds_list, targets_list, average='macro')
    metrics = {'valid_loss': valid_loss, 'valid_acc': valid_acc, 'valid_f1': valid_f1}
    return model, metrics

