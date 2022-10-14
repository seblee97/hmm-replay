from config_manager import config_field, config_template
from hmm_replay import constants


def get_template():
    template_class = HMMConfigTemplate()
    return template_class.base_template


class HMMConfigTemplate:
    def __init__(self):
        self._initialise()

    def _initialise(self):
        self._ode_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.TIMESTEP,
                    types=[float, int],
                    requirements=[lambda x: x > 0],
                ),
            ],
            level=[constants.ODE_RUN],
            dependent_variables=[constants.ODE_SIMULATION],
            dependent_variables_required_values=[[True]],
        )

        self._task_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.TEACHER_CONFIGURATION,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [
                            constants.ROTATION,
                        ]
                    ],
                ),
                config_field.Field(
                    name=constants.NUM_TEACHERS,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
            ],
            level=[constants.TASK],
        )

        self._training_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.TOTAL_TRAINING_STEPS,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.TRAIN_BATCH_SIZE,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.LEARNING_RATE,
                    types=[float, int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.LOSS_FUNCTION,
                    types=[str],
                    requirements=[lambda x: x in [constants.MSE, constants.BCE]],
                ),
                config_field.Field(
                    name=constants.SCALE_HEAD_LR,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.SCALE_HIDDEN_LR,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.TRAIN_HIDDEN_LAYERS,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.TRAIN_HEAD_LAYER,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.FREEZE_FEATURES,
                    types=[list],
                    requirements=[lambda x: all(isinstance(y, int) for y in x)],
                ),
            ],
            level=[constants.TRAINING],
        )

        self._iid_gaussian_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.MEAN,
                    types=[float, int],
                ),
                config_field.Field(
                    name=constants.VARIANCE,
                    types=[int, float],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.DATASET_SIZE,
                    types=[str, int],
                    requirements=[lambda x: x == constants.INF or x > 0],
                ),
            ],
            level=[constants.DATA, constants.IID_GAUSSIAN],
            dependent_variables=[constants.INPUT_SOURCE],
            dependent_variables_required_values=[[constants.IID_GAUSSIAN]],
        )

        self._data_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.INPUT_SOURCE,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [
                            constants.IID_GAUSSIAN,
                        ]
                    ],
                )
            ],
            nested_templates=[self._iid_gaussian_template],
            level=[constants.DATA],
        )

        self._logging_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.VERBOSE,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.LOG_FREQUENCY,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.CHECKPOINT_FREQUENCY,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.PRINT_FREQUENCY,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.SAVE_WEIGHT_FREQUENCY,
                    types=[int, type(None)],
                    requirements=[lambda x: x is None or x > 0],
                ),
                config_field.Field(
                    name=constants.LOG_TO_DF,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.MERGE_AT_CHECKPOINT,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.SAVE_WEIGHTS_AT_SWITCH,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.SAVE_INITIAL_WEIGHTS,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.SAVE_TEACHER_WEIGHTS,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.LOG_OVERLAPS,
                    types=[bool],
                ),
            ],
            level=[constants.LOGGING],
        )

        self._testing_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.TEST_BATCH_SIZE,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.TEST_FREQUENCY,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.OVERLAP_FREQUENCY,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
            ],
            level=[constants.TESTING],
        )

        self._student_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.TEACHER_FEATURES_COPY,
                    types=[type(None), int],
                    requirements=[lambda x: x is None or x >= 0],
                ),
                config_field.Field(
                    name=constants.STUDENT_HIDDEN_LAYERS,
                    types=[list],
                    requirements=[lambda x: all(isinstance(y, int) and y > 0 for y in x)],
                ),
                config_field.Field(
                    name=constants.STUDENT_NONLINEARITY,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [constants.SCALED_ERF, constants.RELU, constants.LINEAR]
                    ],
                ),
                config_field.Field(
                    name=constants.APPLY_NONLINEARITY_ON_OUTPUT,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.STUDENT_INITIALISATION_STD,
                    types=[float, int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.INITIALISE_STUDENT_OUTPUTS,
                    types=[bool],
                ),
                config_field.Field(name=constants.COPY_HEAD_AT_SWITCH, types=[bool]),
                config_field.Field(
                    name=constants.STUDENT_BIAS_PARAMETERS,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.SCALE_STUDENT_FORWARD_BY_HIDDEN,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.SYMMETRIC_STUDENT_INITIALISATION,
                    types=[bool],
                ),
            ],
            level=[constants.MODEL, constants.STUDENT],
        )

        self._rotation_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.READOUT_ROTATION_ALPHA,
                    types=[float, int],
                    requirements=[lambda x: x >= 0 and x <= 1],
                ),
                config_field.Field(
                    name=constants.FEATURE_ROTATION_ALPHA,
                    types=[int, float],
                    requirements=[lambda x: x >= 0 and x <= 1],
                ),
            ],
            level=[
                constants.MODEL,
                constants.TEACHERS,
                constants.ROTATION,
            ],
            dependent_variables=[constants.TEACHER_CONFIGURATION],
            dependent_variables_required_values=[[constants.ROTATION]],
        )

        self._teachers_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.TEACHER_NOISES,
                    types=[list],
                    requirements=[lambda x: all(y >= 0 for y in x)],
                ),
                config_field.Field(
                    name=constants.TEACHER_HIDDEN_LAYERS,
                    types=[list],
                    requirements=[lambda x: all(isinstance(y, int) and y > 0 for y in x)],
                ),
                config_field.Field(
                    name=constants.TEACHER_NONLINEARITIES,
                    types=[list],
                    requirements=[
                        lambda x: all(
                            y in [constants.SCALED_ERF, constants.RELU, constants.LINEAR]
                            for y in x
                        )
                    ],
                ),
                config_field.Field(
                    name=constants.NORMALISE_TEACHERS,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.TEACHER_INITIALISATION_STD,
                    types=[float, int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.UNIT_NORM_TEACHER_HEAD,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.TEACHER_BIAS_PARAMETERS,
                    types=[bool],
                ),
                config_field.Field(
                    name=constants.SCALE_TEACHER_FORWARD_BY_HIDDEN,
                    types=[bool],
                ),
            ],
            nested_templates=[
                self._rotation_template,
            ],
            level=[constants.MODEL, constants.TEACHERS],
        )

        self._model_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.INPUT_DIMENSION,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.OUTPUT_DIMENSION,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
            ],
            nested_templates=[self._teachers_template, self._student_template],
            level=[constants.MODEL],
        )

        self._curriculum_template = config_template.Template(
            fields=[
                config_field.Field(
                    name=constants.STOPPING_CONDITION,
                    types=[str],
                    requirements=[
                        lambda x: x
                        in [
                            constants.FIXED_PERIOD,
                            constants.LOSS_THRESHOLDS,
                            constants.SWITCH_STEPS,
                        ]
                    ],
                ),
                config_field.Field(
                    name=constants.FIXED_PERIOD,
                    types=[int],
                    requirements=[lambda x: x > 0],
                ),
                config_field.Field(
                    name=constants.SWITCH_STEPS,
                    types=[list],
                    requirements=[lambda x: all(isinstance(y, int) and y >= 0 for y in x)],
                ),
                config_field.Field(
                    name=constants.LOSS_THRESHOLDS,
                    types=[list],
                    requirements=[
                        lambda x: all(
                            (isinstance(y, int) or isinstance(y, float)) and y > 0
                            for y in x
                        )
                    ],
                ),
                config_field.Field(
                    name=constants.INTERLEAVE_PERIOD, types=[int, type(None)]
                ),
                config_field.Field(
                    name=constants.INTERLEAVE_DURATION, types=[int, type(None)]
                ),
            ],
            level=[constants.CURRICULUM],
        )

    @property
    def base_template(self):
        return config_template.Template(
            fields=[
                config_field.Field(
                name=constants.EXPERIMENT_NAME,
                types=[str, type(None)],
            ),
            config_field.Field(
                name=constants.USE_GPU,
                types=[bool],
            ),
            config_field.Field(
                name=constants.GPU_ID,
                types=[int],
            ),
            config_field.Field(
                name=constants.SEED,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.RESULTS_PATH,
                types=[str, type(None)],
            ),
            config_field.Field(
                name=constants.NETWORK_SIMULATION,
                types=[bool],
            ),
            config_field.Field(
                name=constants.ODE_SIMULATION,
                types=[bool],
            ),
            ],
            nested_templates=[
                self._ode_template,
                self._task_template,
                self._training_template,
                self._data_template,
                self._logging_template,
                self._testing_template,
                self._model_template,
                self._curriculum_template,
            ],
        )
