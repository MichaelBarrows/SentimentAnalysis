import generate_baseline_results
import generate_initial_experimental_results
import create_next_experiments
import generate_further_experimental_results
import results_sorting
import dataset as ds

## Generate the baseline results for unigrams
# generate_baseline_results.negation_handled(ds.negate_output_unigrams, "unigrams")
# generate_baseline_results.negation_not_handled(ds.not_negate_output_unigrams, "unigrams")

## Generate the inital experiment results for unigrams
# generate_initial_experimental_results.negation_handled(ds.negate_output_unigrams, "unigrams")
# generate_initial_experimental_results.negation_not_handled(ds.not_negate_output_unigrams, "unigrams")

## Create the next experiments for unigrams
# create_next_experiments.run(ds.negate_output_unigrams, "negation_handled", "unigrams")
# create_next_experiments.run(ds.not_negate_output_unigrams, "negation_not_handled", "unigrams")

# # Generate the created experiment results for unigrams
# generate_further_experimental_results.process_negation_handled_experiments(ds.negate_output_unigrams, "unigrams")
# generate_further_experimental_results.process_negation_not_handled_experiments(ds.not_negate_output_unigrams, "unigrams")

# Sort and store best results in separate file
results_sorting.import_results(ds.negate_output_unigrams)
results_sorting.import_results(ds.not_negate_output_unigrams)
results_sorting.import_best_results_and_sort(ds.negate_output_unigrams)
results_sorting.import_best_results_and_sort(ds.not_negate_output_unigrams)
