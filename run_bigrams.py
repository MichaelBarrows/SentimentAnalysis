import generate_baseline_results
import generate_initial_experimental_results
import create_next_experiments
import generate_further_experimental_results
import results_sorting
import dataset as ds

# Generate the baseline results for bigrams
generate_baseline_results.negation_handled(ds.negate_output_bigrams, "bigrams")
generate_baseline_results.negation_not_handled(ds.not_negate_output_bigrams, "bigrams")

# Generate the inital experiment results for bigrams
generate_initial_experimental_results.negation_handled(ds.negate_output_bigrams, "bigrams")
generate_initial_experimental_results.negation_not_handled(ds.not_negate_output_bigrams, "bigrams")

# Create the next experiments for bigrams
create_next_experiments.get_existing_results(ds.negate_output_bigrams, "negation_handled", "bigrams")
create_next_experiments.get_existing_results(ds.not_negate_output_bigrams, "negation_not_handled", "bigrams")

# Generate the created experiment results for bigrams
generate_further_experimental_results.process_negation_handled_experiments(ds.negate_output_bigrams, "bigrams")
generate_further_experimental_results.process_negation_not_handled_experiments(ds.not_negate_output_bigrams, "bigrams")

# Sort and store best results in separate file
results_sorting.import_results(ds.negate_output_bigrams)
results_sorting.import_results(ds.not_negate_output_bigrams)
results_sorting.import_best_results_and_sort(ds.negate_output_bigrams)
results_sorting.import_best_results_and_sort(ds.not_negate_output_bigrams)
