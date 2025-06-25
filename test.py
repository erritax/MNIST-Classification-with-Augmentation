import torch
import torch.nn.functional as F

def test(model, device, test_loader):

    model.eval()

    # counters
    test_loss = 0
    correct = 0 

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # calculate accuracy
    accuracy = correct / len(test_loader.dataset)

    # calculate average loss
    test_loss /= len(test_loader.dataset)

    # output test results
    print(f'\nTest Results:\n'
          f'Accuracy: {accuracy} '
          f'({100. * accuracy:.0f}%)\n'
          f'Average Loss: {test_loss:.4f}\n')
    
    return accuracy
