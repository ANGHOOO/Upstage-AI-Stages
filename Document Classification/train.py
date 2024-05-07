import torch
import tqdm
from sklearn.metrics import accuracy_score, f1_score

def train(model, train_loader, device, criterion, optimizer, mixup_fn=None):
    model.train()
    train_loss = 0.0
    preds_list = []
    targets_list = []
    tbar = tqdm.tqdm(train_loader)
    for batch in tbar:
        images, labels = batch
        if images is None or labels is None:  # skip corrupted images
            continue
        images = images.to(device)
        labels = labels.to(device)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if mixup_fn is None:
            preds_list.extend(outputs.argmax(dim=1).cpu().numpy())
            targets_list.extend(labels.cpu().numpy())
        tbar.set_description(f"Train Loss : {loss.item():.4f}")
    train_loss = train_loss / len(train_loader)
    train_acc = accuracy_score(targets_list, preds_list) if len(targets_list) > 0 else 0
    train_f1 = f1_score(targets_list, preds_list, average='macro') if len(targets_list) > 0 else 0
    metrics = {'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1}
    return model, metrics

def train_loop(model, train_loader, valid_loader, device, criterion, optimizer, scheduler, mixup_fn, num_epochs, patience):
    best_valid_loss = float('inf')
    early_stop_counter = 0
    for epoch in range(num_epochs):
        model, train_metrics = train(model, train_loader, device, criterion, optimizer, mixup_fn)
        model, valid_metrics = evaluate(model, valid_loader, device, criterion, epoch, num_epochs)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {train_metrics['train_loss']:.4f}, Valid Loss: {valid_metrics['valid_loss']:.4f}")
        if valid_metrics['valid_loss'] < best_valid_loss:
            best_valid_loss = valid_metrics['valid_loss']
            early_stop_counter = 0
            torch.save(model.state_dict(), f"{model_path}/model_{model_name}.pt")
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            print('Early Stopping!')        
            break
    return model

