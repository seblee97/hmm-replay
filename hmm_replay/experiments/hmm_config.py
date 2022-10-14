from typing import Dict, List, Union

import yaml
from config_manager import base_configuration
from hmm_replay.experiments import hmm_config_template


class HMMConfig(base_configuration.BaseConfiguration):
    """HMM Wrapper for base configuration

    Implements a specific validate configuration method for
    non-trivial associations that need checking in config.
    """

    def __init__(self, config: Union[str, Dict], changes: List[Dict] = []) -> None:
        base_template = hmm_config_template.get_template()
        super().__init__(
            configuration=config,
            template=base_template,
            changes=changes,
        )

        self._validate_config()

    def _validate_config(self) -> None:
        """Check for non-trivial associations in config.

        Raises:
            AssertionError: if any rules are broken by config.
        """
        pass

    def _maybe_reconfigure(self, property_name: str) -> None:
        pass
