from __future__ import annotations
from pydantic import BaseModel
import json
import os
import random
import numpy as np
import pandas as pd
from sklearn import linear_model

from typing import Optional, Dict, List, Sequence, Any, Literal, cast
from collections import defaultdict
from observatory_utils.general import (
    Subject,
    get_subject_config,
    ActivationRecord,
    ActivationSign,
)
from observatory_utils.exemplar import (
    ExemplarConfig,
    ExemplarSplit,
    ExemplarsWrapper,
    ExemplarType,
    NeuronExemplars,
)
from observatory_utils.simulator import (
    NeuronSimulator,
    ScoredSequenceSimulation,
    calibrate_and_score_simulation,
    correlation_score,
    rsquared_score_from_sequences,
    absolute_dev_explained_score_from_sequences,
    SequenceSimulation,
)


class ExplanationSimulations(BaseModel):
    """Result of scoring a single explanation on multiple sequences."""

    simulation_data: dict[int, ScoredSequenceSimulation]
    """ScoredSequenceSimulation for each sequence"""
    ev_correlation_score: Optional[float] = None
    """
    Correlation coefficient between the expected values of the normalized activations from the
    simulation and the unnormalized true activations on a dataset created from all score_results.
    (Note that this is not equivalent to averaging across sequences.)
    """
    rsquared_score: Optional[float] = None
    """R^2 of the simulated activations."""
    absolute_dev_explained_score: Optional[float] = None
    """
    Score based on absolute difference between real and simulated activations.
    absolute_dev_explained_score = 1 - mean(abs(real-predicted))/ mean(abs(real)).
    """

    def get_preferred_score(self) -> Optional[float]:
        """
        This method may return None in cases where the score is undefined, for example if the
        normalized activations were all zero, yielding a correlation coefficient of NaN.
        """
        return self.ev_correlation_score


class NeuronExplanation(BaseModel):
    """Simulator parameters and the results of scoring it on multiple sequences"""

    explanation: str
    """The explanation used for simulation."""

    simulations: Optional[dict[ExemplarSplit, ExplanationSimulations]] = None
    """Result of scoring the neuron simulator on multiple sequences."""

    def get_preferred_score(
        self, exemplar_splits: Sequence[ExemplarSplit]
    ) -> Optional[float]:
        """
        This method may return None in cases where the score is undefined, for example if the
        normalized activations were all zero, yielding a correlation coefficient of NaN.
        """
        if self.simulations is None:
            return None
        true_activations: List[List[float]] = []
        flattened_sim_activations: List[float] = []
        seq_sims: List[SequenceSimulation] = []
        for split in exemplar_splits:
            if split not in self.simulations:
                return None
            for _, scored_seq_sim in self.simulations[split].simulation_data.items():
                true_activations.append(scored_seq_sim.true_activations)
                uncalibrated_seq_sim = scored_seq_sim.simulation.uncalibrated_simulation
                assert uncalibrated_seq_sim is not None
                flattened_sim_activations.extend(
                    uncalibrated_seq_sim.expected_activations
                )
                seq_sims.append(uncalibrated_seq_sim)
        if not seq_sims:
            return None

        flattened_true_activations = np.concatenate(true_activations)
        # Fit a linear model that maps simulated activations to true activations.
        regression_model = linear_model.LinearRegression()
        regression_model.fit(  # type: ignore
            np.array(flattened_sim_activations).reshape(-1, 1),
            flattened_true_activations,
        )

        scored_seq_sims: dict[int, ScoredSequenceSimulation] = {}
        for exemplar_idx, seq_sim in enumerate(seq_sims):
            scored_seq_sims[exemplar_idx] = calibrate_and_score_simulation(
                seq_sim, true_activations[exemplar_idx], regression_model
            )
        expl_sims = aggregate_scored_sequence_simulations(scored_seq_sims)
        return expl_sims.get_preferred_score()

    def parse_simulation_results(
        self,
        exemplar_split: ExemplarSplit,
        calibration_strategy: Optional[Literal["linreg", "norm"]] = "norm",
    ) -> pd.DataFrame | None:
        if self.simulations is None:
            return None
        simulations = self.simulations[exemplar_split]

        results: List[dict[str, Any]] = []
        for i, scored_simulation in simulations.simulation_data.items():
            score = scored_simulation.ev_correlation_score
            rank = i
            simulation = scored_simulation.simulation

            tokens = simulation.tokens
            true_activations = scored_simulation.true_activations

            if calibration_strategy is None:
                assert simulation.uncalibrated_simulation is not None
                simulated_activations = (
                    simulation.uncalibrated_simulation.expected_activations
                )
            elif calibration_strategy == "linreg":
                simulated_activations = simulation.expected_activations
            elif calibration_strategy == "norm":
                assert simulation.uncalibrated_simulation is not None
                simulated_activations = np.array(
                    simulation.uncalibrated_simulation.expected_activations
                )
                simulated_activations = (
                    simulated_activations / 10 * max(true_activations)
                )

            results.append(
                {
                    "score": score,
                    "rank": rank,
                    "tokens": tokens,
                    "simulated_activations": simulated_activations,
                    "true_activations": true_activations,
                }
            )
        return pd.DataFrame(results)


class ExplanationConfig(BaseModel):
    exemplar_config: ExemplarConfig

    exem_slice_for_exp: tuple[int, int, int] = (0, 20, 1)
    permute_exemplars_for_exp: bool = True
    num_exem_for_exp: Optional[int] = None  # Legacy
    num_exem_range_for_exp: Optional[tuple[int, int]] = (10, 20)
    fix_exemplars_for_exp: bool = True
    permute_examples_for_exp: bool = True
    num_examples_for_exp: Optional[int] = 1
    fix_examples_for_exp: bool = True
    explainer_model_name: str = "gpt-4o"
    add_special_tokens_for_explainer: bool = True
    explainer_system_prompt_type: str = "no_cot"
    use_puzzle_for_bills: bool = False
    examples_placement: str = "fewshot"
    min_tokens_to_highlight: int = 3
    round_to_int: bool = True
    num_explanation_samples: int = 1
    max_new_tokens_for_explanation_generation: int = 2000
    temperature_for_explanation_generation: float = 1.0
    save_full_explainer_responses: bool = False

    exem_slice_to_score: tuple[int, int, int] = (0, 20, 1)
    num_random_seqs_to_score: int = 5
    simulator_model_name: str = "meta-llama/Llama-3.1-70B-Instruct"
    simulator_system_prompt_type: str = "unk_base"
    add_special_tokens: bool = False

    seed: int = 54

    def __str__(self):
        return json.dumps(self.model_dump(), indent=4)


class SplitExemplars(BaseModel):
    split: ExemplarSplit
    neuron_exemplars: NeuronExemplars
    exem_idxs: List[int]

    class Config:
        arbitrary_types_allowed = True

    def get_activation_records(
        self,
        normalize: bool = False,
        mask_opposite_sign: bool = False,
        add_ranks: bool = False,
    ) -> Dict[ExemplarType, List[ActivationRecord] | Dict[int, ActivationRecord]]:
        if normalize:
            act_recs = self.neuron_exemplars.get_normalized_act_records(
                self.split, mask_opposite_sign
            )
        else:
            act_recs = self.neuron_exemplars.activation_records[self.split]

        if add_ranks:
            return {
                extype: {idx: act_recs[extype][idx] for idx in self.exem_idxs}
                for extype in ExemplarType
            }
        else:
            return {
                extype: [act_recs[extype][idx] for idx in self.exem_idxs]
                for extype in ExemplarType
            }

    def get_activation_percentiles(self) -> Dict[float, float]:
        return self.neuron_exemplars.activation_percentiles

    def get_ranks(self) -> List[int]:
        return self.exem_idxs


def aggregate_scored_sequence_simulations(
    scored_sequence_simulations: Dict[int, ScoredSequenceSimulation],
) -> ExplanationSimulations:
    """
    Aggregate a list of scored sequence simulations. The logic for doing this is non-trivial for EV
    scores, since we want to calculate the correlation over all activations from all sequences at
    once rather than simply averaging per-sequence correlations.
    """
    all_true_activations: list[float] = []
    all_expected_values: list[float] = []
    for _, scored_sequence_simulation in scored_sequence_simulations.items():
        all_true_activations.extend(scored_sequence_simulation.true_activations or [])
        all_expected_values.extend(
            scored_sequence_simulation.simulation.expected_activations
        )
    ev_correlation_score = (
        correlation_score(all_true_activations, all_expected_values)
        if len(all_true_activations) > 0
        else None
    )
    rsquared_score = rsquared_score_from_sequences(
        all_true_activations, all_expected_values
    )
    absolute_dev_explained_score = absolute_dev_explained_score_from_sequences(
        all_true_activations, all_expected_values
    )

    return ExplanationSimulations(
        simulation_data=scored_sequence_simulations,
        ev_correlation_score=ev_correlation_score,
        rsquared_score=rsquared_score,
        absolute_dev_explained_score=absolute_dev_explained_score,
    )


def filter_activations(
    activations: List[float], min_or_max: ExemplarType
) -> List[float]:
    if min_or_max == ExemplarType.MAX:
        return np.maximum(activations, 0).tolist()
    else:
        return np.abs(np.minimum(activations, 0)).tolist()


def simulate_and_score(
    split_exemplars: SplitExemplars,
    explanations: List[NeuronExplanation],
    exemplar_type: ExemplarType,
    simulator: NeuronSimulator,
    overwrite: bool = False,
) -> List[NeuronExplanation]:
    exemplars = cast(
        Dict[int, ActivationRecord],
        split_exemplars.get_activation_records(add_ranks=True)[exemplar_type],
    )

    # Go through current data and find which explanations need evaluating on which exemplars.
    explanations_to_eval_per_exemplar: dict[int, List[int]] = defaultdict(
        list
    )  # {exemplar_rank: [expl_idxs]}
    exemplars_to_eval_per_explanation: dict[int, List[int]] = defaultdict(
        list
    )  # {expl_idx: [exemplar_ranks]}
    # Initialize uncalibrated sequence simulations with previous results so that we can
    # use the full set of results for calibration.
    uncalib_seq_sims: dict[int, dict[int, SequenceSimulation]] = defaultdict(
        dict
    )  # {expl_idx: {exemplar_rank: SequenceSimulation}}
    for expl_idx, explanation in enumerate(explanations):
        if explanation.simulations is not None:
            expl_sim_results = explanation.simulations.get(split_exemplars.split, None)
        else:
            expl_sim_results = None

        exemplar_ranks_to_eval = set(exemplars.keys())
        if not overwrite and expl_sim_results is not None:
            already_evaluated_exemplar_ranks = set(
                expl_sim_results.simulation_data.keys()
            )
            exemplar_ranks_to_eval = (
                exemplar_ranks_to_eval - already_evaluated_exemplar_ranks
            )
            for (
                exemplar_rank,
                scored_seq_sim,
            ) in expl_sim_results.simulation_data.items():
                uncalib_sim = scored_seq_sim.simulation.uncalibrated_simulation
                assert uncalib_sim is not None
                uncalib_seq_sims[expl_idx][exemplar_rank] = uncalib_sim

        for exemplar_rank in exemplar_ranks_to_eval:
            explanations_to_eval_per_exemplar[exemplar_rank].append(expl_idx)
            exemplars_to_eval_per_explanation[expl_idx].append(exemplar_rank)

    # If there are no more things to evaluate, return.
    if not exemplars_to_eval_per_explanation:
        return explanations

    # Run simulations for explanations and exemplars that we haven't evaluated before.
    for exemplar_rank, expl_idxs in explanations_to_eval_per_exemplar.items():
        explanation_strs = [explanations[idx].explanation for idx in expl_idxs]
        act_rec = exemplars[exemplar_rank]
        sim_results_list = simulator.simulate(
            explanations=explanation_strs,
            tokens=act_rec.tokens,
            token_ids=act_rec.token_ids,
        )

        for expl_idx, sim_results in zip(expl_idxs, sim_results_list):
            uncalib_seq_sims[expl_idx][exemplar_rank] = sim_results

    # Get true and simulated activations for calibration.
    exemplar_ranks = sorted(explanations_to_eval_per_exemplar.keys())
    true_activations = {
        exemplar_rank: filter_activations(
            exemplars[exemplar_rank].activations, exemplar_type
        )
        for exemplar_rank in exemplar_ranks
    }
    flattened_true_activations = np.concatenate(
        [true_activations[exemplar_rank] for exemplar_rank in exemplar_ranks]
    )
    simulated_activations: Dict[int, Dict[int, List[float]]] = defaultdict(
        dict
    )  # {expl_idx: {exemplar_rank: simulated activations}}
    for expl_idx in uncalib_seq_sims:
        for exemplar_rank, uncalib_seq_sim in uncalib_seq_sims[expl_idx].items():
            simulated_activations[expl_idx][
                exemplar_rank
            ] = uncalib_seq_sim.expected_activations

    results: List[NeuronExplanation] = []
    for expl_idx, explanation in enumerate(explanations):
        if expl_idx not in exemplars_to_eval_per_explanation:
            assert not overwrite and explanation.simulations is not None
            results.append(explanation)
            continue

        flattened_simulated_activations = np.concatenate(
            [
                simulated_activations[expl_idx][exemplar_rank]
                for exemplar_rank in exemplar_ranks
            ]
        )
        # Fit a linear model that maps simulated activations to true activations.
        regression_model = linear_model.LinearRegression()
        regression_model.fit(  # type: ignore
            flattened_simulated_activations.reshape(-1, 1), flattened_true_activations
        )

        # Maybe initialize from old simulation_data.
        full_sim_results = (
            explanation.simulations.copy()
            if explanation.simulations is not None
            else {}
        )
        scored_sequence_simulations: Dict[int, ScoredSequenceSimulation] = (
            {}
        )  # {exemplar_rank: ScoredSequenceSimulation}
        for exemplar_rank, sim_results in uncalib_seq_sims[expl_idx].items():
            scored_seq_sim = calibrate_and_score_simulation(
                sim_results, true_activations[exemplar_rank], regression_model
            )
            scored_sequence_simulations[exemplar_rank] = scored_seq_sim

        expl_sims = aggregate_scored_sequence_simulations(scored_sequence_simulations)
        full_sim_results[split_exemplars.split] = expl_sims
        results.append(
            NeuronExplanation(
                explanation=explanation.explanation,
                simulations=full_sim_results,
            )
        )
    return results


class ExplanationsWrapper:
    def __init__(
        self,
        exemplar_data_dir: str,
        config: ExplanationConfig,
        subject: Subject,
        overwrite: bool = False,
    ):
        exemplars_wrapper = ExemplarsWrapper(
            data_dir=exemplar_data_dir,
            config=config.exemplar_config,
            subject=subject,
        )

        # os.makedirs(save_path, exist_ok=True)
        # for layer in range(subject.L):
        #     os.makedirs(
        #         os.path.join(save_path, "explanations", str(layer)), exist_ok=True
        #     )

        # # Check whether data already exists in save_path.
        # config_file = os.path.join(save_path, "explanation_config.json")
        # if not overwrite and os.path.exists(config_file):
        #     # Check that the configs are the same (excluding some fields that can be different).
        #     with open(config_file, "r") as f:
        #         existing_config = ExplanationConfig.model_validate_json(f.read())
        #     fields_to_exclude = set(["save_full_explainer_responses"])
        #     existing_config_dict = existing_config.model_dump(exclude=fields_to_exclude)
        #     config_dict = config.model_dump(exclude=fields_to_exclude)
        #     for field in existing_config_dict:
        #         existing_val = existing_config_dict[field]
        #         curr_val = config_dict[field]
        #         if existing_val != curr_val:
        #             print(
        #                 f"Value of '{field}' for existing config is '{existing_val}', "
        #                 f"while the value given in the config is '{curr_val}')",
        #                 f"Existing config file: {config_file}",
        #                 f"New config file: {config_file}",
        #             )
        # else:
        #     # If there's no data saved yet, save our config.
        #     with open(config_file, "w") as f:
        #         f.write(config.model_dump_json())

        # Parse train/valid/test split idxs.
        exem_indices_for_exp = list(range(*config.exem_slice_for_exp))
        exem_indices_to_score = list(range(*config.exem_slice_to_score))

        # Support for legacy config.
        # TODO(damichoi): Remove this at some point.
        if config.num_exem_for_exp is not None:
            config.num_exem_range_for_exp = (
                config.num_exem_for_exp,
                config.num_exem_for_exp,
            )

        # Don't initialize explainer until we need it, since it might use GPU memory.
        self.explainer = None

        # Keep track of randomness using generator.
        self.rng = random.Random(config.seed)

        # self.base_save_path = save_path
        self.config = config
        self.exemplars_wrapper = exemplars_wrapper
        self.exem_indices_for_exp = exem_indices_for_exp
        self.exem_indices_to_score = exem_indices_to_score

    def get_split_neuron_exemplars(
        self,
        to_score: bool,
        split: ExemplarSplit,
        layer: int,
        neuron_idx: int,
        # custom_vectors_save_name: str | None = None,
    ) -> SplitExemplars:
        neuron_exemplars = self.exemplars_wrapper.get_neuron_exemplars(
            layer,
            neuron_idx,
            # custom_vectors_save_name=custom_vectors_save_name,
        )
        if split.startswith("random"):
            exem_indices = list(range(self.config.num_random_seqs_to_score))
        else:
            exem_indices = (
                self.exem_indices_for_exp
                if not to_score
                else self.exem_indices_to_score
            )
        return SplitExemplars(
            split=split, neuron_exemplars=neuron_exemplars, exem_idxs=exem_indices
        )

    def score_arbitrary_explanation(
        self,
        explanation: str,
        layer: int,
        neuron_idx: int,
        act_sign: ActivationSign,
        simulator: NeuronSimulator,
        exem_splits: Sequence[ExemplarSplit] = (
            ExemplarSplit.VALID,
            ExemplarSplit.RANDOM_VALID,
        ),
        # custom_vectors_save_name: str | None = None,
    ) -> NeuronExplanation:
        neuron_exemplars = self.exemplars_wrapper.get_neuron_exemplars(
            layer,
            neuron_idx,
            # custom_vectors_save_name=custom_vectors_save_name,
        )
        explanations = [NeuronExplanation(explanation=explanation)]
        for split in exem_splits:
            if split.startswith("random"):
                exem_indices = list(range(self.config.num_random_seqs_to_score))
            else:
                exem_indices = self.exem_indices_to_score
            split_exemplars = SplitExemplars(
                split=split, neuron_exemplars=neuron_exemplars, exem_idxs=exem_indices
            )

            extype = (
                ExemplarType.MAX if act_sign == ActivationSign.POS else ExemplarType.MIN
            )
            scored_explanations = simulate_and_score(
                split_exemplars=split_exemplars,
                explanations=explanations,
                exemplar_type=extype,
                simulator=simulator,
                overwrite=True,
            )
            explanations = scored_explanations
        return explanations[0]
