import torch
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    
    model.train()
    epoch_loss = 0
    print(f'Train Epoch: {epoch}')

    # loop through batches
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # output progress after each batch
        if batch_idx % log_interval == 0:
            print(f'[{batch_idx * len(data):>5}/{len(train_loader.dataset)}'
                    f'({100. * batch_idx / len(train_loader):>2.0f}%)]\t'
                    f'Loss: {loss.item():.6f}')

            torch.save(model.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')
        
        avg_loss = epoch_loss / len(train_loader)
        return avg_loss