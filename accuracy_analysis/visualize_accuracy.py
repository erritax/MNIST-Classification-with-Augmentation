import json
import matplotlib.pyplot as plt

# load accuracy results from results dir
with open('results/accuracy_real.json', 'r') as f:
    acc_real = json.load(f)

with open('results/accuracy_with_synthetic.json', 'r') as f:
    acc_synth = json.load(f)

# prepare data
sizes = sorted([int(k) for k in acc_real.keys()])
acc_real_list = [acc_real[str(k)] for k in sizes]
acc_synth_list = [acc_synth[str(k)] for k in sizes]

# plot data
plt.figure(figsize=(12, 6))
plt.plot(sizes, acc_real_list, color = 'cornflowerblue',
         marker = 'o', label = 'Real Only')
plt.plot(sizes, acc_synth_list, color = 'magenta',
         marker = 'o', label = 'Real + Synthetic')
plt.title('Classification Accuracy vs. Dataset Size')
plt.xlabel('# Real Training Samples')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True)
plt.show()