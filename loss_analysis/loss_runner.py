from loss_analysis.evaluate_loss import run_evaluation, save_loss

dataset_sizes = [100, 1000, 5000, 10000, 25000, 50000]

# evaluate losses with only real data
_, loss_real = run_evaluation(dataset_sizes, use_synthetic=False)
save_loss(loss_real, 'loss_real')

# evaluate losses with real and synthetic data
_, loss_synth = run_evaluation(dataset_sizes, use_synthetic=True)
save_loss(loss_synth, 'loss_with_synthetic')