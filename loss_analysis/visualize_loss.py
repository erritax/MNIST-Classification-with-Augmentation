import json
import matplotlib.pyplot as plt

# load data
with open('results/loss_real.json', 'r') as f:
    loss_real = json.load(f)

with open('results/loss_with_synthetic.json', 'r') as f:
    loss_synth = json.load(f)

dataset_sizes = ['100', '1000', '5000', '10000', '25000', '50000']
colors = ['orchid', 'teal', 'salmon', 'yellowgreen', 'orange', 'indigo']

epochs = list(range(1, len(next(iter(loss_real.values()))) + 1))

plt.figure(figsize=(12, 6))

# plot
for i, size in enumerate(dataset_sizes):
    if size in loss_real:
        plt.plot(epochs, loss_real[size], marker = 'o', 
                 color = colors[i], label = f'{size}')

    if size in loss_synth:
        plt.plot(epochs, loss_synth[size], marker = 'o', 
                 color = colors[i], linestyle = '--')

plt.title('Average Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.ylim(0, 2.5)
plt.grid(True)
plt.tight_layout()
plt.legend(title="Dataset Size", fontsize = 8)
plt.figtext(0.05, 0.01, 
            "Solid lines = Real Data\nDashed lines = Real + Synthetic Data", 
            wrap=True, horizontalalignment = 'left', fontsize = 8)
plt.show()