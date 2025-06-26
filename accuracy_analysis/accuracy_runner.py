from accuracy_analysis.evaluate_accuracy import run_evaluation, save_results

dataset_sizes = [100, 250, 500, 1000, 5000, 10000, 25000, 50000]

# evaluate accuracy with only real data
results_real = run_evaluation(dataset_sizes, use_synthetic=False)
save_results(results_real, 'accuracy_real')

# evaluate accuracy with real and synthetic data
results_synth = run_evaluation(dataset_sizes, use_synthetic=True)
save_results(results_synth, 'accuracy_with_synthetic')