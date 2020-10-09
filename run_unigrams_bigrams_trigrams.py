import generate_baseline_results
import generate_initial_experimental_results
import create_next_experiments
import generate_further_experimental_results
import results_sorting
import dataset as ds

# execute_negated_dataset()
# parameters:
#   None
# returns:
#   None
# description:
#   This function calls and executes all code linked to the negation handled
#       dataset for unigrams, bigrams & trigrams
def execute_negated_dataset ():
    # Generate the baseline results for unigrams, bigrams & trigrams
    generate_baseline_results.negation_handled(ds.negate_output_unigrams_bigrams_trigrams, "unigrams_bigrams_trigrams")
    # Generate the inital experiment results for unigrams, bigrams & trigrams
    generate_initial_experimental_results.negation_handled(ds.negate_output_unigrams_bigrams_trigrams, "unigrams_bigrams_trigrams")
    # Create the next experiments for unigrams, bigrams & trigrams
    create_next_experiments.run(ds.negate_output_unigrams_bigrams_trigrams, "negation_handled", "unigrams_bigrams_trigrams")
    # Generate the created experiment results for unigrams, bigrams & trigrams
    generate_further_experimental_results.process_negation_handled_experiments(ds.negate_output_unigrams_bigrams_trigrams, "unigrams_bigrams_trigrams")
    # Sort and store best results in separate file
    results_sorting.import_results(ds.negate_output_unigrams_bigrams_trigrams)
    results_sorting.import_best_results_and_sort(ds.negate_output_unigrams_bigrams_trigrams)

# execute_negated_dataset()
# parameters:
#   None
# returns:
#   None
# description:
#   This function calls and executes all code linked to the negation not handled
#       dataset for unigrams, bigrams & trigrams
def execute_not_negated_dataset ():
    # Generate the baseline results for unigrams, bigrams & trigrams
    generate_baseline_results.negation_not_handled(ds.not_negate_output_unigrams_bigrams_trigrams, "unigrams_bigrams_trigrams")
    # Generate the inital experiment results for unigrams, bigrams & trigrams
    generate_initial_experimental_results.negation_not_handled(ds.not_negate_output_unigrams_bigrams_trigrams, "unigrams_bigrams_trigrams")
    # Create the next experiments for unigrams, bigrams & trigrams
    create_next_experiments.run(ds.not_negate_output_unigrams_bigrams_trigrams, "negation_not_handled", "unigrams_bigrams_trigrams")
    # Generate the created experiment results for unigrams, bigrams & trigrams
    generate_further_experimental_results.process_negation_not_handled_experiments(ds.not_negate_output_unigrams_bigrams_trigrams, "unigrams_bigrams_trigrams")
    # Sort and store best results in separate file
    results_sorting.import_results(ds.not_negate_output_unigrams_bigrams_trigrams)
    results_sorting.import_best_results_and_sort(ds.not_negate_output_unigrams_bigrams_trigrams)

execute_negated_dataset()
execute_not_negated_dataset()
