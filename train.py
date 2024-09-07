from tqdm import tqdm

def train_model(model, train_dataloader, n_epochs, optimizer, device):
    train_losses = []
    
    model.to(device)
    
    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        # Training loop
        for data, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            data = data.to(device)
            optimizer.zero_grad()
            encoded, decoded = model(data)
            
            loss = model.loss(data, decoded, encoded)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        print(f'\nEpoch: {epoch+1}/{n_epochs} -- Train Loss: {avg_train_loss:.4f}')
        print('-'*50)

    return train_losses

