def prepare_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
    )
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1))

    return train_dataset, test_dataset


# Modified BCEWithLogitsLoss with Label Smoothing
class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(BCEWithLogitsLossWithSmoothing, self).__init__()
        self.smoothing = smoothing
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce_loss(inputs, targets)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, early_stopping_patience=6):
    model.to(device)
    scaler = GradScaler()
    accumulation_steps = 8

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'f1_train': [], 'f1_val': [], 'auc_train': [], 'auc_val': []}

    best_f1 = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        y_true_train = []
        y_pred_train = []
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_seq = inputs.unsqueeze(1)

            with autocast():
                outputs = model(inputs_seq, inputs_seq)
                loss = criterion(outputs.squeeze(-1), labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (preds.squeeze() == labels.squeeze()).sum().item()
            total_train += labels.size(0)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(preds.squeeze().cpu().numpy())

        train_loss /= total_train
        train_acc = correct_train / total_train
        f1_train = f1_score(y_true_train, y_pred_train)
        auc_train = roc_auc_score(y_true_train, y_pred_train)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['f1_train'].append(f1_train)
        history['auc_train'].append(auc_train)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.unsqueeze(1), inputs.unsqueeze(1))
                loss = criterion(outputs.squeeze(-1), labels)

                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (preds.squeeze() == labels.squeeze()).sum().item()
                total_val += labels.size(0)

                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.squeeze().cpu().numpy())

        val_loss /= total_val
        val_acc = correct_val / total_val
        f1_val = f1_score(y_true_val, y_pred_val)
        auc_val = roc_auc_score(y_true_val, y_pred_val)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['f1_val'].append(f1_val)
        history['auc_val'].append(auc_val)

        scheduler.step(val_loss)


        if f1_val > best_f1:
            best_f1 = f1_val
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return history


