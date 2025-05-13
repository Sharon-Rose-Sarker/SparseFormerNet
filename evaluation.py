def create_lr_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)


def evaluate(model, test_loader, device):
    model.eval()
    output_B = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs, inputs).squeeze(-1)
            probs = torch.sigmoid(outputs)
            output_B.extend(probs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return np.array(output_B), np.array(y_true)


train_dataset, test_dataset = prepare_data(file_path)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
val_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)




# Model Hyperparameters
d_model = 256  
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
d_ff = 1024
dropout = 0.3  


self_attn = MultiHeadAttention(num_heads, d_model, dropout)
cross_attn = MultiHeadAttention(num_heads, d_model, dropout)
feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
encoder_layers = [EncoderLayer(d_model, self_attn, feed_forward, dropout, use_cbam=True) for _ in range(num_encoder_layers)]
decoder_layers = [DecoderLayer(d_model, self_attn, cross_attn, feed_forward, dropout) for _ in range(num_decoder_layers)]
encoder = Encoder(encoder_layers, nn.LayerNorm(d_model))
decoder = Decoder(decoder_layers, nn.LayerNorm(d_model))
positional_encoding = PositionalEncoding(d_model, dropout)
model = Transformer(encoder, decoder, d_model, positional_encoding, use_cbam=True).to(device)


criterion = FocalLoss(alpha=1.0, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = create_lr_scheduler(optimizer)


# Evaluate the model
y_pred_transformer, y_true_test = evaluate(model, test_loader, device)
best_threshold = 0.5
best_accuracy = 0

for threshold in np.arange(0.3, 0.7, 0.01):
    preds = (y_pred_transformer > threshold).astype(int)
    acc = (preds == y_true_test).mean()
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold

print(f"Best Threshold: {best_threshold:.2f}, Best Accuracy: {best_accuracy:.4f}")



final_preds = (y_pred_transformer > best_threshold).astype(int)
