import copy
import os
import time
from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from hmm_replay import constants
from hmm_replay.curricula import (base_curriculum, hard_steps_curriculum,
                                  periodic_curriculum, threshold_curriculum)
from hmm_replay.data_modules import base_data_module, iid_data
from hmm_replay.experiments import hmm_config
from hmm_replay.networks import network_configuration, student
from hmm_replay.networks.teacher_ensembles import (base_teacher_ensemble,
                                                   rotation_ensemble)
from hmm_replay.utils import decorators
from run_modes import base_runner


class NetworkRunner(base_runner.BaseRunner):
    """Runner for network simulations.

    Class for orchestrating student teacher framework including training
    and test loops.
    """

    def __init__(
        self, config: hmm_config.HMMConfig, unique_id: str = ""
    ) -> None:
        """
        Class constructor.

        Args:
            config: configuration object containing parameters to specify run.
        """
        # extract class-relevant attributes from config
        self._seed = config.seed
        self._device = config.experiment_device
        self._input_dimension = config.input_dimension
        self._checkpoint_frequency = config.checkpoint_frequency
        self._print_frequency = config.print_frequency
        self._checkpoint_path = config.checkpoint_path
        self._save_weight_frequency = config.save_weight_frequency
        self._total_training_steps = config.total_training_steps
        self._log_frequency = config.log_frequency
        self._test_frequency = config.test_frequency
        self._total_step_count = 0
        self._log_overlaps = config.log_overlaps
        self._overlap_frequency = config.overlap_frequency

        # initialise student, teachers, logger_module,
        # data_module, loss_module, torch optimiser, and curriculum object
        self._teachers = self._setup_teachers(config=config)
        self._num_teachers = len(self._teachers.teachers)
        self._student = self._setup_student(config=config)
        # self._logger = self._setup_logger(config=config)
        self._data_module = self._setup_data(config=config)
        self._loss_function = self._setup_loss(config=config)
        self._optimiser = self._setup_optimiser(config=config)
        self._curriculum = self._setup_curriculum(config=config)

        self._manage_network_devices()

        super().__init__(config=config, unique_id=unique_id)

    def _get_data_columns(self):
        columns = [constants.TEACHER_INDEX]
        generalisation_error_tags = [
            f"{constants.GENERALISATION_ERROR}_{i}" for i in range(self._num_teachers)
        ]
        log_generalisation_error_tags = [
            f"{constants.LOG_GENERALISATION_ERROR}_{i}"
            for i in range(self._num_teachers)
        ]
        columns.extend(generalisation_error_tags)
        columns.extend(log_generalisation_error_tags)
        columns.append(constants.LOSS)
        if self._log_overlaps:
            sample_network_config = self.get_network_configuration()
            columns.extend(list(sample_network_config.dictionary.keys()))
    
        return columns

    def get_network_configuration(self) -> network_configuration.NetworkConfiguration:
        """Get macroscopic configuration of networks in terms of order parameters.

        Used for both logging purposes and as input to ODE runner.
        """
        with torch.no_grad():
            student_head_weights = [
                head.weight.data.cpu().numpy().flatten() for head in self._student.heads
            ]
            teacher_head_weights = [
                teacher.head.weight.data.cpu().numpy().flatten()
                for teacher in self._teachers.teachers
            ]
            student_self_overlap = self._student.self_overlap.cpu().numpy()
            teacher_self_overlaps = [
                teacher.self_overlap.cpu().numpy()
                for teacher in self._teachers.teachers
            ]
            teacher_cross_overlaps = [
                o.cpu().numpy() for o in self._teachers.cross_overlaps
            ]
            student_layer = self._student.layers[0].weight.data
            student_teacher_overlaps = [
                student_layer.mm(teacher.layers[0].weight.data.t()).cpu().numpy()
                / self._input_dimension
                for teacher in self._teachers.teachers
            ]
            
            student_old_student_overlap = []

        return network_configuration.NetworkConfiguration(
            student_head_weights=student_head_weights,
            teacher_head_weights=teacher_head_weights,
            student_self_overlap=student_self_overlap,
            teacher_self_overlaps=teacher_self_overlaps,
            teacher_cross_overlaps=teacher_cross_overlaps,
            student_teacher_overlaps=student_teacher_overlaps,
            student_old_student_overlap=student_old_student_overlap,
        )

    @decorators.timer
    def _setup_teachers(
        self, config: hmm_config.HMMConfig
    ) -> base_teacher_ensemble.BaseTeacherEnsemble:
        """Initialise teacher object containing teacher networks."""
        forward_scaling = (
            1 / np.sqrt(config.teacher_hidden_layers[0])
            if config.scale_teacher_forward_by_hidden
            else 1.0
        )
        base_arguments = {
            constants.INPUT_DIMENSION: config.input_dimension,
            constants.HIDDEN_DIMENSIONS: config.teacher_hidden_layers,
            constants.OUTPUT_DIMENSION: config.output_dimension,
            constants.BIAS: config.teacher_bias_parameters,
            constants.NUM_TEACHERS: config.num_teachers,
            constants.NONLINEARITIES: config.teacher_nonlinearities,
            constants.SCALE_HIDDEN_LR: config.scale_hidden_lr,
            constants.FORWARD_SCALING: forward_scaling,
            constants.UNIT_NORM_TEACHER_HEAD: config.unit_norm_teacher_head,
            constants.WEIGHT_NORMALISATION: config.normalise_teachers,
            constants.NOISE_STDS: config.teacher_noises,
            constants.INITIALISATION_STD: config.teacher_initialisation_std,
        }
        if config.teacher_configuration == constants.ROTATION:
            teachers_class = rotation_ensemble.RotationTeacherEnsemble
            additional_arguments = {
                constants.FEATURE_ROTATION_ALPHA: config.feature_rotation_alpha,
                constants.READOUT_ROTATION_ALPHA: config.readout_rotation_alpha,
            }
        else:
            raise ValueError(
                f"Teacher configuration '{config.teacher_configuration}' not recognised."
            )

        teachers = teachers_class(**base_arguments, **additional_arguments)

        if config.save_teacher_weights:
            save_path = os.path.join(
                config.checkpoint_path, constants.TEACHER_WEIGHT_SAVE_PATH
            )
            teachers.save_all_teacher_weights(save_path=save_path)

        return teachers

    @decorators.timer
    def _setup_student(
        self, config: hmm_config.HMMConfig
    ) -> student.Student:
        """Initialise object containing student network."""

        teacher_features_copy = (
            None
            if config.teacher_features_copy is None
            else copy.deepcopy(
                self._teachers.teachers[config.teacher_features_copy].layers
            )
        )

        return student.Student(
            input_dimension=config.input_dimension,
            hidden_dimensions=config.student_hidden_layers,
            output_dimension=config.output_dimension,
            bias=config.student_bias_parameters,
            num_teachers=config.num_teachers,
            learning_rate=config.learning_rate,
            scale_head_lr=config.scale_head_lr,
            scale_hidden_lr=config.scale_hidden_lr,
            scale_forward_by_hidden=config.scale_student_forward_by_hidden,
            nonlinearity=config.student_nonlinearity,
            freeze_features=config.freeze_features,
            train_hidden_layers=config.train_hidden_layers,
            train_head_layer=config.train_head_layer,
            initialise_outputs=config.initialise_student_outputs,
            copy_head_at_switch=config.copy_head_at_switch,
            apply_nonlinearity_on_output=config.apply_nonlinearity_on_output,
            symmetric_initialisation=config.symmetric_student_initialisation,
            initialisation_std=config.student_initialisation_std,
            teacher_features_copy=teacher_features_copy,
        )

    @decorators.timer
    def _setup_data(
        self, config: hmm_config.HMMConfig
    ) -> base_data_module.BaseData:
        """Initialise data module."""
        if config.input_source == constants.IID_GAUSSIAN:
            data_module = iid_data.IIDData(
                train_batch_size=config.train_batch_size,
                test_batch_size=config.test_batch_size,
                input_dimension=config.input_dimension,
                mean=config.mean,
                variance=config.variance,
                dataset_size=config.dataset_size,
            )
        else:
            raise ValueError(
                f"Data module (specified by input source) {config.input_source} not recognised"
            )
        return data_module

    @decorators.timer
    def _setup_loss(
        self, config: hmm_config.HMMConfig
    ) -> Callable:
        """Instantiate torch loss function"""
        if config.loss_function == "mse":
            loss_function = nn.MSELoss()
        elif config.loss_function == "bce":
            loss_function = nn.BCELoss()
        else:
            raise NotImplementedError(
                f"Loss function {config.loss_function} not recognised."
            )
        return loss_function

    @decorators.timer
    def _setup_curriculum(
        self, config: hmm_config.HMMConfig
    ) -> base_curriculum.BaseCurriculum:
        """Initialise curriculum object (when to switch teacher,
        how to decide subsequent teacher etc.)

        Raises:
            ValueError: if stopping condition is not recognised.
        """
        if config.stopping_condition == constants.FIXED_PERIOD:
            curriculum = periodic_curriculum.PeriodicCurriculum(config=config)
        elif config.stopping_condition == constants.LOSS_THRESHOLDS:
            curriculum = threshold_curriculum.ThresholdCurriculum(config=config)
        elif config.stopping_condition == constants.SWITCH_STEPS:
            curriculum = hard_steps_curriculum.HardStepsCurriculum(config=config)
        else:
            raise ValueError(
                f"Stopping condition {config.stopping_condition} not recognised."
            )
        return curriculum

    @decorators.timer
    def _setup_optimiser(
        self, config: hmm_config.HMMConfig
    ) -> torch.optim.SGD:
        """Initialise optimiser with trainable parameters of student."""
        trainable_parameters = self._student.get_trainable_parameters()
        return torch.optim.SGD(trainable_parameters, lr=config.learning_rate)

    @decorators.timer
    def _manage_network_devices(self) -> None:
        """Move relevant networks etc. to device specified in config (CPU or GPU)."""
        self._student.to(device=self._device)
        for teacher in self._teachers.teachers:
            teacher.to(device=self._device)

    def _compute_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate loss of prediction of student vs. target from teacher

        Args:
            prediction: prediction made by student network on given input
            target: teacher output on same input

        Returns:
            loss: loss between target (from teacher) and prediction (from student)
        """
        loss = 0.5 * self._loss_function(prediction, target)
        return loss

    @decorators.timer
    def _setup_training(self):
        """Prepare runner for training, including constructing a test dataset.
        This method must be called before training loop is called.
        """
        # different configurations make different number of calls to rng
        # reset seeds before training to ensure data is the same.
        # experiment_utils.set_random_seeds(self._seed)

        self._test_data_inputs = self._data_module.get_test_data()[constants.X].to(
            self._device
        )
        self._test_teacher_outputs = self._teachers.forward_all(self._test_data_inputs)

    def _pretrain_logging(self):
        """Log relevant quantities before start of training."""
        pretrain_logging_dict = self._compute_generalisation_errors()
        if self._log_overlaps:
            network_config = self.get_network_configuration()
            pretrain_logging_dict = {
                **pretrain_logging_dict,
                **network_config.dictionary,
            }
        self._log_step_data(step=0, logging_dict=pretrain_logging_dict)

        if self._save_weight_frequency is not None:
            if self._total_step_count % self._save_weight_frequency == 0:
                self._student.save_weights(
                    save_path=os.path.join(
                        self._checkpoint_path,
                        f"{constants.STUDENT_WEIGHTS}_{self._total_step_count}",
                    )
                )

        # for i, error in enumerate(generalisation_errors):
        #     self._data_logger.write_scalar(
        #         tag=f"{constants.GENERALISATION_ERROR}_{i}",
        #         step=0,
        #         scalar=error,
        #     )
        #     self._data_logger.write_scalar(
        #         tag=f"{constants.LOG_GENERALISATION_ERROR}_{i}",
        #         step=0,
        #         scalar=np.log10(error),
        #     )

    def train(self):
        """Training orchestration."""

        self._setup_training()
        self._pretrain_logging()

        while self._total_step_count <= self._total_training_steps:
            teacher_index = next(self._curriculum)

            self._train_on_teacher(teacher_index=teacher_index)

            first_task = False

            self._student.save_weights(
                save_path=os.path.join(
                    self._checkpoint_path,
                    f"{constants.STUDENT_WEIGHTS}_{self._total_step_count}",
                )
            )

        # import pdb

        # pdb.set_trace()

        # self._data_logger.checkpoint()

    def _train_on_teacher(self, teacher_index: int):
        """One phase of training (wrt one teacher)."""
        self._student.signal_task_boundary(new_task=teacher_index)

        task_step_count = 0
        latest_generalisation_errors = [np.inf for _ in range(self._num_teachers)]

        timer = time.time()

        while self._total_step_count <= self._total_training_steps:

            if (
                self._total_step_count % self._checkpoint_frequency == 0
                and self._total_step_count != 0
            ):
                self._data_logger.checkpoint()

            self._total_step_count += 1
            task_step_count += 1

            step_logging_dict = self._train_test_step(
                teacher_index=teacher_index,
            )

            latest_generalisation_errors = [
                step_logging_dict.get(
                    f"{constants.GENERALISATION_ERROR}_{i}",
                    latest_generalisation_errors[i],
                )
                for i in range(self._num_teachers)
            ]

            if self._total_step_count % self._print_frequency == 0:
                if self._total_step_count != 0:
                    self._logger.info(f"Time for last {self._print_frequency} steps: {time.time() - timer}")
                    timer = time.time()
                self._logger.info(
                    f"Generalisation errors @ (~) step {self._total_step_count} "
                    f"({task_step_count}'th step training on teacher {teacher_index}): "
                )
                for i in range(self._num_teachers):
                    self._logger.info(
                        f"    Teacher {i}: {latest_generalisation_errors[i]}\n"
                    )

            if self._save_weight_frequency is not None:
                if self._total_step_count % self._save_weight_frequency == 0:
                    self._student.save_weights(
                        save_path=os.path.join(
                            self._checkpoint_path,
                            f"{constants.STUDENT_WEIGHTS}_{self._total_step_count}",
                        )
                    )

            if self._curriculum.to_switch(
                task_step=task_step_count,
                error=latest_generalisation_errors[teacher_index],
            ):
                if self._log_overlaps:
                    if len(self._curriculum.history) < 2:
                        network_config = self.get_network_configuration()
                        step_logging_dict = {
                            **step_logging_dict,
                            **network_config.dictionary,
                        }
                    self._log_step_data(
                        step=self._total_step_count, logging_dict=step_logging_dict
                    )
                break

            self._log_step_data(
                step=self._total_step_count, logging_dict=step_logging_dict
            )

    def _train_test_step(
        self, teacher_index: int
    ) -> Dict[str, Any]:
        step_logging_dict = self._training_step(
            teacher_index=teacher_index
        )

        self._student.signal_step(step=self._total_step_count)

        if self._total_step_count % self._test_frequency == 0:
            generalisation_errors = self._compute_generalisation_errors()
            step_logging_dict = {**step_logging_dict, **generalisation_errors}

        if self._log_overlaps and self._total_step_count % self._overlap_frequency == 0:
            network_config = self.get_network_configuration()
            step_logging_dict = {**step_logging_dict, **network_config.dictionary}

        step_logging_dict[constants.TEACHER_INDEX] = teacher_index

        return step_logging_dict

    def _training_step(
        self, teacher_index: int
    ):
        """Perform single training step."""

        training_step_dict = {}

        batch = self._data_module.get_batch()
        batch_input = batch[constants.X].to(self._device)

        # forward through student network
        student_output = self._student.forward(batch_input)

        # forward through teacher network(s)
        teacher_output = self._teachers.forward(teacher_index, batch_input)

        # training iteration
        self._optimiser.zero_grad()
        loss = self._compute_loss(student_output, teacher_output)

        training_step_dict[constants.LOSS] = loss.item()

        loss.backward()
        self._optimiser.step()

        return training_step_dict

    def _compute_generalisation_errors(self) -> List[float]:
        """Compute test errors for student with respect to all teachers."""
        self._student.eval()

        generalisation_errors = {}

        with torch.no_grad():
            student_outputs = self._student.forward_all(self._test_data_inputs)

            # meta student will only have one set of outputs from forward_all call
            if len(student_outputs) == 1:
                student_outputs = [
                    student_outputs[0] for _ in range(len(self._test_teacher_outputs))
                ]

            for i, (student_output, teacher_output) in enumerate(
                zip(student_outputs, self._test_teacher_outputs)
            ):
                loss = self._compute_loss(student_output, teacher_output)
                generalisation_errors[
                    f"{constants.GENERALISATION_ERROR}_{i}"
                ] = loss.item()
                generalisation_errors[
                    f"{constants.LOG_GENERALISATION_ERROR}_{i}"
                ] = np.log10(loss.item())

        self._student.train()

        return generalisation_errors

    def _log_step_data(self, step: int, logging_dict: Dict[str, Any]):
        for tag, scalar in logging_dict.items():
            self._data_logger.write_scalar(tag=tag, step=step, scalar=scalar)

    def post_process(self):
        self._plotter.load_data()
        self._plotter.plot_learning_curves()
