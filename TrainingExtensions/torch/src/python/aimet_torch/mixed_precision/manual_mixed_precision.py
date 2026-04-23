# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# /usr/bin/env python

"""Manual mixed precision configurator"""

from typing import overload, Union, List, Tuple, Dict, get_args, Type, Optional, IO
import torch

from aimet_torch.common.quantsim_config.json_config_importer import JsonConfigImporter
from aimet_torch.common.utils import AimetLogger
from aimet_torch.utils import flatten_list
from aimet_torch.mixed_precision.utils import (
    UserRequest,
    RequestType,
    SupportedDType,
    ModuleProduct,
    broadcast_tuples,
)
from aimet_torch.mixed_precision.manual_mixed_precision_handler import MpHandler
from aimet_torch.quantsim import QuantizationSimModel

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)


class MixedPrecisionConfigurator:
    """
    Mixed Precision Configurator helps set up a mixed precision profile in the QuantSim object. The user is expected to
    follow the below steps to set the sim in Mixed Precision.

    1. Create QuantSim object
    2. Create the MixedPrecisionConfigurator object by passing in the QuantSim object
    3. Make a series of set_precision/set_model_input_precision/set_model_output_precision calls
    4. Call apply() method by passing in the config file and strict flag
    5. Run compute_encodings on the above QuantSim object
    6. Export the encodings/onnx artifacts
    """

    def __init__(self, sim: QuantizationSimModel):
        """
        :param sim: QuantSim object
        """
        self._sim = sim
        self.user_requests = []
        # pylint: disable=protected-access
        self.mp_handler = MpHandler(
            sim, JsonConfigImporter.import_json_config_file(self._sim._config_file)
        )

    def _store_user_request(
        self,
        request_type: RequestType,
        module: Union[torch.nn.Module, Type, ModuleProduct],
        activation: Union[List[SupportedDType], SupportedDType] = None,
        param: Optional[Dict[str, SupportedDType]] = None,
    ):
        self.user_requests.append(
            UserRequest(
                request_type=request_type,
                module=module,
                activation=activation,
                param=param,
            )
        )

    @overload
    def set_precision(
        self,
        module: torch.nn.Module,
        activation: Union[List[SupportedDType], SupportedDType],
        param: Optional[Dict[str, SupportedDType]] = None,
    ): ...

    @overload
    def set_precision(
        self,
        module_type: Type[torch.nn.Module],
        activation: Union[List[SupportedDType], SupportedDType],
        param: Optional[Dict[str, SupportedDType]] = None,
    ): ...

    def set_precision(
        self,
        arg: Union[torch.nn.Module, Type[torch.nn.Module]],
        activation: Union[List[SupportedDType], SupportedDType],
        param: Optional[Dict[str, SupportedDType]] = None,
    ):
        """
        :param arg: Module can be of type torch.nn.Module or the type of the module.
        :param activation: A string representing the activation dtype of the module input(s)
        :param param: Dict with name of the param as key and its dtype as value

        - If the 'module' is a leaf-module(the module doesnt compose of other torch.nn.module), the specified settings
          would be applied to the module.
        - If the 'module' is a non-leaf-module (module is composed of other torch.nn.module), the specified settings
          would be applied to all the leaf modules in 'module'.
        - If the 'module' is Type of module, all the modules in the model which satisfy the specified module type would
          be set to the specified activation and param settings
        - If the same 'module' is specified through multiple set_precision(...) calls, the latest one will be applied.

        Examples: TODO

        """
        # pylint: disable=too-many-branches
        if activation:
            if isinstance(activation, List):
                for act in activation:
                    if act not in get_args(SupportedDType):
                        raise ValueError(
                            "Supported inputs for activation are ",
                            get_args(SupportedDType),
                        )
            else:
                if activation not in get_args(SupportedDType):
                    raise ValueError(
                        "Supported inputs for activation are ", get_args(SupportedDType)
                    )
        if param:
            for param_name, dtype in param.items():
                if dtype not in get_args(SupportedDType):
                    raise ValueError(
                        f"Supported inputs for param: {param_name} are ",
                        get_args(SupportedDType),
                    )

        if isinstance(arg, type):
            self._store_user_request(
                RequestType.set_precision_by_module_type, arg, activation, param
            )
        elif isinstance(arg, torch.nn.Module):
            if arg in self._sim.model.modules():
                self._store_user_request(
                    RequestType.set_precision_by_module, arg, activation, param
                )
            else:
                raise ValueError(
                    f"Specified module {arg} is not part of the sim object"
                )
        else:
            raise TypeError(
                "arg is neither a torch.nn.Module nor of Type[torch.nn.Module]"
            )

    def set_model_input_precision(
        self,
        activation: Union[
            List[Optional[SupportedDType]],
            Tuple[Optional[SupportedDType]],
            SupportedDType,
        ],
    ):
        """
        Activation precision which needs to be set to the model inputs

        :param activation: Activation dtypes for inputs of the model
        """
        broadcasted_activations = broadcast_tuples(
            activation, self.mp_handler.cg_traverser.model_inputs
        )
        for act, model_input in zip(
            flatten_list(broadcasted_activations),
            flatten_list(self.mp_handler.cg_traverser.model_inputs),
        ):
            if act:
                if act not in get_args(SupportedDType):
                    raise ValueError(
                        "Supported inputs for activation are ", get_args(SupportedDType)
                    )
                self._store_user_request(
                    RequestType.set_model_input_precision, model_input, act
                )

    def set_model_output_precision(
        self,
        activation: Union[
            List[Optional[SupportedDType]],
            Tuple[Optional[SupportedDType]],
            SupportedDType,
        ],
    ):
        """
        Activation precision which needs to be set to the model outputs

        :param activation: Activation dtypes for outputs of the model
        """
        broadcasted_activations = broadcast_tuples(
            activation, self.mp_handler.cg_traverser.model_outputs
        )
        for act, model_output in zip(
            flatten_list(broadcasted_activations),
            flatten_list(self.mp_handler.cg_traverser.model_outputs),
        ):
            if act:
                if act not in get_args(SupportedDType):
                    raise ValueError(
                        "Supported inputs for activation are ", get_args(SupportedDType)
                    )
                self._store_user_request(
                    RequestType.set_model_output_precision, model_output, act
                )

    @overload
    def apply(self, log_file: str = "./mmp_log.txt", strict: bool = True): ...

    @overload
    def apply(self, log_file: IO, strict: bool = True): ...

    def apply(
        self, log_file: Optional[Union[IO, str]] = "./mmp_log.txt", strict: bool = True
    ):
        """
        Apply the mp settings specified through the set_precision/set_model_input_precision/set_model_output_precision
        calls to the QuantSim object

        :param log_file: log_file to store the logs. log_file can either be a string representing the path or the IO
                        object to write the logs into.
        :param strict: Boolean flag to indicate whether to fail (strict=True) on incorrect/conflicting inputs made by
                       the user or (strict=False) take a best-effort approach to realize the MP settings
        """
        if isinstance(log_file, str):
            with open(log_file, "w") as f:
                self.mp_handler.apply(f, self.user_requests, strict)
        else:
            self.mp_handler.apply(log_file, self.user_requests, strict)

        self.user_requests.clear()
