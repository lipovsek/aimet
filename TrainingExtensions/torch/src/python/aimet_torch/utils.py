# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=too-many-lines, redefined-builtin
"""Utilities that are used for different AIMET PyTorch features"""

import itertools
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Callable,
    Any,
    Iterable,
    Optional,
    TextIO,
    Mapping,
)
import contextlib
from contextlib import contextmanager, ExitStack
import os
import pickle
import logging
import functools
from packaging import version

import torch.nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils._pytree import tree_map
from torch.nn.modules.module import (
    _global_backward_hooks,
    _global_forward_pre_hooks,
    _global_forward_hooks,
)

try:
    from torch.nn.modules.module import _global_backward_pre_hooks
except ImportError:
    _global_backward_pre_hooks = None

from torchvision import datasets, transforms

from aimet_torch.common.utils import AimetLogger, Handle, profile as _profile
from aimet_torch.common.quantsim import _get_minimum_scale
from aimet_torch.quantization._utils import interleave, concretize_block_size


logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Utils)

dtypes_to_ignore_for_quantization = (int, bool, str, tuple, type(None))
torch_dtypes_to_ignore_for_quantization = [
    torch.int,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.uint8,
]
allowed_output_types = (torch.Tensor, float, *dtypes_to_ignore_for_quantization)
DROPOUT_TYPES = (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)

# list of modules which need to be treated as a leaf module
modules_to_treat_as_leaf = []

# list of modules not to treat as leaf
modules_not_to_treat_as_leaf = [torch.nn.ModuleList, torch.nn.ModuleDict]


class StopForwardException(Exception):
    """
    Dummy exception to early-terminate forward-pass
    """


class ModuleData:
    """
    Collect input and output data to and from module
    """

    def __init__(
        self,
        model: torch.nn.Module,
        module: torch.nn.Module,
        forward_fn: Callable[[torch.nn.Module, Any], Any] = None,
    ):
        """
        :param model: Pytorch model
        :param module: Module reference
        :param forward_fn: Adapter function that performs forward pass given a model and inputs
         yielded from the data loader.
        """
        self._model = model
        self._module = module
        self._forward_fn = forward_fn or self.default_forward_fn

    def collect_inp_out_data(
        self, args, kwargs: Mapping[str, Any], collect_input: bool, collect_output: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect input and output data depending on the collect_input and collect_output flag

        :param model_input: Input to model, Can be a single tensor or a list/tuple of tensors
        :param collect_input: Boolean to collect input or not
        :param collect_output: Boolean to collect output or not
        :return: Module's input and output data
        """

        def adjust_input_dtype(module, inp):
            if hasattr(module, "weight") and module.weight is not None:
                dtype = module.weight.dtype
                # Cast input to dtype only if it is a floating point tensor (float, half, bfloat16, etc.).
                # If input is a non-float tensor (e.g. long, bool), leave the input uncasted.
                return tree_map(
                    lambda x: x.to(dtype)
                    if isinstance(x, torch.Tensor) and x.is_floating_point()
                    else x,
                    inp,
                )
            return inp

        handles = [
            mod.register_forward_pre_hook(adjust_input_dtype)
            for mod in self._model.modules()
        ]

        def _hook_to_collect_inp_out_data(_, inp, out):
            """
            hook to collect input and output data
            """
            if collect_input:
                inp_data_list.append(inp[0])

            if collect_output:
                out_data_list.append(out)

            raise StopForwardException

        inp_data_list = []
        out_data_list = []

        handles.append(
            self._module.register_forward_hook(_hook_to_collect_inp_out_data)
        )

        # get the model's device placement information
        device = get_device(self._model)

        # place the input to appropriate device
        args = change_tensor_device_placement(args, device)
        kwargs = change_tensor_device_placement(kwargs, device)

        # Custom injected exception is raised when the activations data from desired module is collected.
        try:
            with in_eval_mode(self._model), torch.no_grad():
                _ = self._forward_fn(self._model, *args, **kwargs)
        except StopForwardException:
            pass
        finally:
            # remove hook handle
            for handle in handles:
                handle.remove()

        inp_data, out_data = None, None

        if inp_data_list and isinstance(inp_data_list[0], torch.Tensor):
            inp_data = inp_data_list[0].detach()

        if out_data_list and isinstance(out_data_list[0], torch.Tensor):
            out_data = out_data_list[0].detach()

        return inp_data, out_data

    @staticmethod
    def default_forward_fn(
        model: torch.nn.Module,
        inputs: Union[torch.tensor, List[torch.Tensor], Tuple[torch.Tensor]],
    ):
        """
        Default forward function that performs forward pass given a model and inputs yielded from
        the data loader. Data loader which yields torch.Tensor object that can be directly
        passed into the model, or a data loader which yields a tuple of length two where its
        first element can be directly passed into the model.

        :param model: PyTorch model.
        :param inputs: Inputs passed to model.
        """
        # When provided dataloader is labeled (model_inputs, labels), then ignore the second element (labels).
        if isinstance(inputs, (list, tuple)):
            inputs, _ = inputs
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        model(*inputs)


class CachedDataset(Dataset):
    """
    Cache number of batches from the data loader at given path location and
    provide interface to fetch single batch of model inputs.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, data_loader: DataLoader, num_batches: Optional[int], path: str):
        """
        :param data_loader: Data loader
        :param num_batches: Number of batches to fetch from data loader
        :param path: Path to save model inputs
        """
        if data_loader:
            if num_batches is not None and len(data_loader) < num_batches:
                raise ValueError(
                    f"Can not fetch {num_batches} batches from "
                    f"a data loader of length {len(data_loader)}."
                )

            self._num_batches = None
            self._path = path

            if num_batches is None:
                self._cache_model_inputs(data_loader)
            else:
                self._cache_model_inputs(itertools.islice(data_loader, num_batches))

            assert self._num_batches is not None
        else:
            assert len(os.listdir(path)) == num_batches
            self._num_batches = num_batches
            self._path = path
            logger.info(
                "Found %d batches of data at path location: %s",
                self._num_batches,
                self._path,
            )

    def __len__(self):
        return self._num_batches

    def __getitem__(self, index: int):
        path = os.path.join(self._path, "model_inputs_" + str(index))

        with open(path, "rb") as file:
            batch = pickle.load(file)

        return batch

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def _cache_model_inputs(self, data_loader):
        """
        Function to cache number of batches individually in separate file at provided path location
        """
        if not os.path.exists(self._path):
            os.makedirs(self._path)

        for i, batch in enumerate(data_loader):
            path = os.path.join(self._path, f"model_inputs_{i}")
            args = (batch,)
            kwargs = {}
            with open(path, "wb") as file:
                pickle.dump((args, kwargs), file)
            self._num_batches = i + 1

        logger.info(
            "Caching %d batches from data loader at path location: %s",
            self._num_batches,
            self._path,
        )


def run_hook_for_layers(
    model: torch.nn.Module,
    input_shapes: Union[Tuple, List[Tuple]],
    hook,
    module_type_for_attaching_hook=None,
    leaf_node_only=True,
):
    """
    Register the given hook function for all layers in the model
    :param model: Model
    :param input_shapes: Shape of inputs to pass to the model
    :param hook: Hook function to register
    :param module_type_for_attaching_hook: Tuple of torch.nn module types for which hook has to be attached
    :param leaf_node_only: Set to False if all modules are required
    :return: None
    """

    # ------------------------
    # Register hook function
    # ------------------------
    hooks = []
    # All leaf modules
    modules = [
        module
        for module in model.modules()
        if not leaf_node_only or is_leaf_module(module)
    ]
    if module_type_for_attaching_hook:
        # if needed, filter by module types specified by caller
        modules = [
            module
            for module in modules
            if isinstance(module, module_type_for_attaching_hook)
        ]
    for module in modules:
        hooks.append(module.register_forward_hook(hook))

    # ------------------------------------------------
    # Run forward pass to execute the hook functions
    # ------------------------------------------------
    device = get_device(model)
    dummy_tensors = create_rand_tensors_given_shapes(input_shapes, device)
    with in_eval_mode(model), torch.no_grad():
        _ = model(*dummy_tensors)

    # --------------------------
    # Remove all hooks we added
    # --------------------------
    for h in hooks:
        h.remove()


def run_hook_for_layers_with_given_input(
    model: torch.nn.Module,
    input_tensor: Union[torch.Tensor, Tuple],
    hook,
    module_type_for_attaching_hook=None,
    leaf_node_only=True,
    fwd_func=None,
):
    """
    Register the given hook function for all layers in the model
    :param model: Model
    :param input_tensor: Input tensor to the model. If more than one model inputs, use a tuple
    :param hook: Hook function to register
    :param module_type_for_attaching_hook: Tuple of torch.nn module types for which hook has to be attached
    :param leaf_node_only: Set to False if all modules are required
    :param fwd_func: forward function for model inference
    :return: None
    """
    # pylint: disable=too-many-branches
    # ------------------------
    # Register hook function
    # ------------------------
    hooks = []
    # All leaf modules
    modules = []

    # Based on the modules in modules_to_treat_as_leaf, we do not want to further continue searching for next level
    # of modules present in modules_to_treat_as_leaf. To achieve this, save them in modules_to_skip
    modules_to_skip = set()

    for module in model.modules():
        if module not in modules_to_skip:
            # pylint: disable=protected-access
            if isinstance(module, tuple(modules_to_treat_as_leaf)):
                modules.append(module)
                # check for modules inside the 'module' and add them to modules_to_skip
                for sub_module in module._modules.values():
                    modules_to_skip.add(sub_module)
            else:
                if leaf_node_only:
                    if is_leaf_module(module):
                        modules.append(module)
                else:
                    modules.append(module)

    if module_type_for_attaching_hook:
        # if needed, filter by module types specified by caller
        modules = [
            module
            for module in modules
            if isinstance(module, module_type_for_attaching_hook)
        ]

    try:
        for module in modules:
            hooks.append(module.register_forward_hook(hook))

        # ------------------------------------------------
        # Run forward pass to execute the hook functions
        # ------------------------------------------------
        with in_eval_mode(model), torch.no_grad():
            if fwd_func:
                _ = fwd_func(model, input_tensor)
            else:
                if isinstance(input_tensor, (list, tuple)):
                    _ = model(*input_tensor)
                elif isinstance(input_tensor, dict):
                    try:
                        _ = model(**input_tensor)
                    except TypeError:
                        # Some models require inputs as dict.
                        # https://github.com/pytorch/vision/blob/ef2920cc80bac61282b3b19a775b3c33de4e7551/torchvision/ops/feature_pyramid_network.py#L172
                        _ = model(input_tensor)
                else:
                    _ = model(input_tensor)

    finally:
        # --------------------------
        # Remove all hooks we added
        # --------------------------
        for h in hooks:
            h.remove()


def create_fake_data_loader(dataset_size: int, batch_size: int, image_size=(1, 28, 28)):
    """
    Helper function to create fake data loader which is default image size (1, 28, 28)
    :param dataset_size     : total images in data set
    :param batch_size       : batch size
    :param image_size       : size of input
    :return:
    """
    transform = transforms.Compose([transforms.ToTensor()])
    data_loader = torch.utils.data.DataLoader(
        datasets.FakeData(
            size=dataset_size,
            image_size=image_size,
            num_classes=10,
            transform=transform,
            target_transform=None,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return data_loader


def get_module_to_name_dict(
    model: torch.nn.Module, prefix: str = ""
) -> Dict[torch.nn.Module, str]:
    """
    Get a dictionary mapping model modules to names
    :param model: Model to get mapping for
    :param prefix: Prefix string to prepend to names
    :return: Dictionary mapping model modules to names
    """
    module_to_name_dict = {}
    for name, module in model.named_modules(prefix=prefix):
        module_to_name_dict[module] = name
    return module_to_name_dict


def get_layer_name(model, layer):
    """
    Helper function to get layer name given model and layer reference
    :param model: model (nn.Module)
    :param layer: layer reference
    :return:
    """
    for name, module in model.named_modules():
        if module is layer:
            return name
    raise KeyError(f"Couldn't find layer {layer} from model {model}")


def is_model_on_gpu(model):
    """
    Function to check whether given model is created on GPU or CPU
    Assumption : model is on single device
    :return:
        True if the model is on GPU, False if on CPU
    """
    return next(model.parameters()).is_cuda


def get_device(model):
    """
    Function to find which device is model on
    Assumption : model is on single device
    :param model:
    :return: Device on which model is present
    """
    return next(model.parameters()).device


def is_leaf_module(module):
    """Utility function to determine if the given module is a leaf module - that is, does not have children modules
    :return:
        True if the module is a leaf, False otherwise
    """
    # pylint: disable=import-outside-toplevel
    from aimet_torch._base.nn.modules._spconv import CustomSparseConv3DLayer

    try:
        _ = next(module.children())
    except StopIteration:
        has_child = False
    else:
        has_child = True

    return (
        not has_child
        or type(module) in modules_to_treat_as_leaf
        or (
            CustomSparseConv3DLayer is not None
            and isinstance(module, CustomSparseConv3DLayer)
        )
    ) and not isinstance(module, tuple(modules_not_to_treat_as_leaf))


def has_hooks(module: torch.nn.Module):
    """Returns True if the module uses hooks."""
    # pylint: disable=protected-access
    return (
        module._backward_hooks
        or module._backward_pre_hooks
        or module._forward_hooks
        or module._forward_pre_hooks
        or _global_backward_pre_hooks
        or _global_backward_hooks
        or _global_forward_hooks
        or _global_forward_pre_hooks
    )


def get_ordered_list_of_modules(
    model: torch.nn.Module,
    dummy_input: Union[torch.Tensor, List[torch.Tensor], Tuple],
    fwd_func=None,
    ignore_duplicates=False,
) -> List:
    """
    Finds ordered modules in given model.
    :param model: PyTorch model.
    :param dummy_input: Dummy input to the model. Used to parse model graph.
    :param fwd_func: forward function for model inference
    :param ignore_duplicates: If True, don't add a module to ordered_list again if it was already seen before.
    :return: List of module name, module in order.
    """
    seen_modules = set()

    def _hook_to_collect_name_of_module(module, _, __):
        """
        hook to find name of module
        """
        module_name = module_to_name_dict[module]
        if module_name in seen_modules and ignore_duplicates:
            return
        list_modules.append([module_name, module])
        seen_modules.add(module_name)

    module_to_name_dict = {}
    for name, module in model.named_modules():
        module_to_name_dict[module] = name

    list_modules = []
    run_hook_for_layers_with_given_input(
        model, dummy_input, hook=_hook_to_collect_name_of_module, fwd_func=fwd_func
    )

    return list_modules


def replace_modules(
    model: torch.nn.Module,
    condition: Callable[[torch.nn.Module], bool],
    factory: Callable[[torch.nn.Module], torch.nn.Module],
):
    """
    Replace all modules that satisfy the given condition
    """

    def fn(parent):
        for name, child in parent.named_children():
            if condition(child):
                setattr(parent, name, factory(child))

    model.apply(fn)


def create_rand_tensors_given_shapes(input_shape, device: torch.device):
    """
    Given shapes of some tensors, create one or more random tensors and return them as a list of tensors

    :param input_shape: Shapes of tensors to create (possibly nested) tuple of integers
    :param device: Device to create tensors on
    :return: Created list of tensors
    """
    try:
        input_shapes = [torch.Size(input_shape)]
    except TypeError:
        input_shapes = input_shape

    rand_tensors = []
    for shape in input_shapes:
        try:
            t = torch.rand(torch.Size(shape), device=device)
        except TypeError:
            t = create_rand_tensors_given_shapes(shape, device)

        rand_tensors.append(t)

    return rand_tensors


def get_ordered_lists_of_conv_fc(
    model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple, List]
) -> List:
    """
    Finds order of nodes in graph
    :param model: model
    :param dummy_input: A dummy input to the model. Can be a Tensor or a Tuple of Tensors
    :return: List of names in graph in order
    """
    module_list = get_ordered_list_of_modules(model, dummy_input)
    module_list = [
        [name, module]
        for name, module in module_list
        if isinstance(
            module,
            (
                torch.nn.Conv1d,
                torch.nn.Conv2d,
                torch.nn.Linear,
                torch.nn.ConvTranspose2d,
                torch.nn.Conv3d,
            ),
        )
    ]
    return module_list


def change_tensor_device_placement(input_data, device: torch.device):
    """
    Change the tensor_data's device placement

    :param input_data: torch.tensor , list of torch.tensors, or tuple of torch.tensors
    :param device: device
    :return: tensor_data with modified device placement
    """
    return tree_map(
        lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, input_data
    )


def nested_map(data, fn: Callable[[torch.Tensor], torch.Tensor]):
    """
    Apply a function to a nested tuple, list, or dict of tensors.
    :param data: Tensor, or a nested tuple, list, or dict of tensors.
    :param fn: Function to apply to the tensors
    :return: Nested structure of tensors with function applied
    """
    if isinstance(data, torch.Tensor):
        return fn(data)

    if isinstance(data, (tuple, list)):
        cls = tuple if isinstance(data, tuple) else list
        return cls(nested_map(x, fn) for x in data)

    if isinstance(data, dict):
        return {key: nested_map(value, fn) for key, value in data.items()}

    logger.debug(
        "unexpected input type=%s, expecting torch.Tensor, tuple, list, or dict. skipping..",
        type(data),
    )
    return data


def find_num_inout_tensors_per_module(model: torch.nn.Module, input_tensor) -> Dict:
    """
    Returns a map of module -> number of output tensors, for all the children modules of the
    provided module

    :param model: Torch module to find children modules for
    :param input_tensor: Input tensor to use to run forward pass for the model. If model needs more than one input
                         tensor, pass a tuple
    :return: map of module -> number of output tensors
    """

    num_inout_map = {}

    def record_num_outputs(module, inputs, outputs):
        num_inputs = len(inputs) if isinstance(inputs, (List, Tuple)) else 1
        num_outputs = len(outputs) if isinstance(outputs, (List, Tuple)) else 1
        num_inout_map[module] = (num_inputs, num_outputs)

    run_hook_for_layers_with_given_input(model, input_tensor, record_num_outputs)
    return num_inout_map


def get_reused_modules(
    model: torch.nn.Module, model_input: Union[torch.Tensor, Tuple]
) -> List[Tuple[str, torch.nn.Module]]:
    """
    Identify modules which are used more than once in the model
    :param model: Model to check for modules used more than once
    :param model_input: Input to the model
    :return: List of tuples of name and module for modules in the model which are used more than once
    """
    module_set = set()
    reused_modules_set = set()

    def forward_hook(curr_module, _, _1):
        """
        Custom forward hook function to add modules to module_set and reused_module_set.
        :param curr_module: Current module being traversed during forward pass.
        :param _1: Unused param
        """
        if curr_module in module_set:
            reused_modules_set.add(curr_module)
        else:
            module_set.add(curr_module)

    run_hook_for_layers_with_given_input(model, model_input, forward_hook)

    reused_modules_list = []
    for name, module in model.named_modules():
        if is_leaf_module(module) and module in reused_modules_set:
            reused_modules_list.append((name, module))
    return reused_modules_list


@contextlib.contextmanager
def in_eval_mode(module: Union[torch.nn.Module, Iterable[torch.nn.Module]]):
    """
    Utility to temporarily put model in eval mode using context manager.
    :param module: PyTorch module or a list of modules
    :return: None
    """
    with _in_mode(module, train=False):
        yield


@contextlib.contextmanager
def in_train_mode(module: Union[torch.nn.Module, Iterable[torch.nn.Module]]):
    """
    Utility to temporarily put model in train mode using context manager.
    :param module: PyTorch module or a list of modules
    :return: None
    """
    with _in_mode(module, train=True):
        yield


@contextlib.contextmanager
def _in_mode(modules: Union[torch.nn.Module, Iterable[torch.nn.Module]], train: bool):
    if isinstance(modules, torch.nn.Module):
        modules = (modules,)

    modules = set(itertools.chain(*(m.modules() for m in modules)))

    original_modes = {module: module.training for module in modules}

    try:
        for module in modules:
            module.training = train
        yield
    finally:
        for module, original_mode in original_modes.items():
            module.training = original_mode


def is_torch_nn_module(module: torch.nn.Module) -> bool:
    """
    Utility function to determine if the given module is from torch.nn class or not.
    For modules like torch.nn.Conv2d, the utility will return True.

    :param module: PyTorch module.
    :return: True if the module from torch.nn class, False otherwise
    """
    return (
        isinstance(module, torch.nn.Module)
        and type(module) in torch.nn.__dict__.values()
    )


def is_torch_nn_leaf_module(module: torch.nn.Module) -> bool:
    """
    Utility function to determine if the given module is leaf and from torch.nn class or not.
    :param module: PyTorch module.
    :return: True if the module is leaf and from torch.nn class, False otherwise
    """
    torch_nn_leaf_module = False
    if is_leaf_module(module) and is_torch_nn_module(module):
        torch_nn_leaf_module = True
    return torch_nn_leaf_module


def get_torch_tensortype_shape(
    torch_graph_output: torch._C.TensorType,
) -> Union[None, List[int]]:
    """
    Given an output tensor from a torch graph, return its shape, or return None if the output tensor is not a
    tensortype.
    """
    # pylint: disable=protected-access
    shape = None
    if isinstance(torch_graph_output.type(), torch._C.TensorType):
        shape = torch_graph_output.type().sizes()
    return shape


def get_all_quantizers(model: torch.nn.Module):
    """
    Get all the quantizers in the model
    :param model: Root module
    :returns: List of parameter, input, and output quantizers
    """
    param_quantizers = []
    input_quantizers = []
    output_quantizers = []

    for module in model.modules():
        _param_qtzrs = getattr(module, "param_quantizers", {}).values()
        _input_qtzrs = getattr(module, "input_quantizers", [])
        _output_qtzrs = getattr(module, "output_quantizers", [])

        if _param_qtzrs:
            param_quantizers.extend(_param_qtzrs)

        if _input_qtzrs:
            input_quantizers.extend(
                _input_qtzrs.values()
                if isinstance(_input_qtzrs, dict)
                else _input_qtzrs
            )

        if _output_qtzrs:
            output_quantizers.extend(
                _output_qtzrs.values()
                if isinstance(_output_qtzrs, dict)
                else _output_qtzrs
            )

    return param_quantizers, input_quantizers, output_quantizers


def disable_all_quantizers(model: torch.nn.Module):
    """
    Temporarily disable all quantizers in the model within with-as block, or permanently disable
    without employing context manager.

    :param model: Root module
    :returns: Handle that enable all quantizers in the model upon handle.remove().
    """
    # pylint: disable=import-outside-toplevel, cyclic-import
    from aimet_torch.nn.base import BaseQuantizationMixin

    if any(isinstance(m, BaseQuantizationMixin) for m in model.modules()):
        return remove_all_quantizers(model)

    param_quantizers, input_quantizers, output_quantizers = get_all_quantizers(model)
    all_quantizers = param_quantizers + input_quantizers + output_quantizers

    active_quantizers = set(
        quantizer for quantizer in all_quantizers if quantizer.enabled
    )

    def cleanup():
        for quantizer in active_quantizers:
            quantizer.enabled = True

    try:
        for quantizer in active_quantizers:
            quantizer.enabled = False
        return Handle(cleanup)
    except:
        cleanup()
        raise


def save_to_cache(tensor, dir_path, idx):
    """
    Save tensor data into provided path with index
    :param tensor: Tensor
    :param dir_path: Provided path to save data
    :param idx: Index of the file
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    path = os.path.join(dir_path, f"model_inputs_{idx}")
    with open(path, "wb") as cache:
        pickle.dump(tensor, cache)


def cache_intermediate_datasets(
    cached_dataset,
    cache_on_cpu,
    model,
    module_name,
    forward_fn,
    path=None,
    incl_kwargs: bool = False,
):
    """
    Cache the input tensor of the target module and save to CPU or disk for latter usage
    :param cached_dataset: Cached dataset
    :param cache_on_cpu: True if caching data on CPU, False if caching to disk
    :param model: Model that contains the target module
    :param module_name: Name of the target module
    :param forward_fn: Forward function that performs forward pass given a model and inputs
    :param path: Location to save cached data if caching to dick
    :param incl_kwargs: if True, capture kwargs, normalize and attach to inputs.
    :return: Cached data on CPU
    """
    # pylint: disable=cell-var-from-loop, too-many-locals, missing-class-docstring, missing-function-docstring
    cached_data = []
    *parent_name, child_name = module_name.split(".")
    parent = model.get_submodule(".".join(parent_name))
    orig_child = getattr(parent, child_name)

    class CachingHelper(torch.nn.Module):
        def forward(self, *args, **kwargs):
            if not incl_kwargs:
                kwargs = {}

            if cache_on_cpu:
                cached_data.append(
                    change_tensor_device_placement((args, kwargs), torch.device("cpu"))
                )
            else:
                save_to_cache((args, kwargs), path, idx)

            raise StopForwardException

    caching_helper = CachingHelper()

    try:
        setattr(parent, child_name, caching_helper)

        iterator = iter(cached_dataset)
        for idx in range(len(cached_dataset)):
            args, kwargs = next(iterator)
            try:
                with in_eval_mode(model), torch.no_grad():
                    _ = forward_fn(model, *args, **kwargs)
            except StopForwardException:
                pass

        return cached_data
    finally:
        setattr(parent, child_name, orig_child)


def profile(
    label: str,
    file: Union[str, os.PathLike, TextIO] = None,
    new_file: bool = False,
    logger: Optional[logging.Logger] = None,  # pylint: disable=redefined-outer-name
):
    """
    Profile a block of code and save profiling information into a file.

    :param label: String label associated with the block of code to profile (shows up in the profiling print)
    :param file: File path and name or a file-like object to send output text to (Default: stdout)
    :param new_file: True if a new file is to be created to hold profiling info, False if an existing file should be
        appended to. This flag is only valid when ``file`` is a path, not a file-like object.
    :param logger: If logger is provided, profiling string will also be printed with INFO logging level
    """
    if not torch.cuda.is_available():
        return profile_async(label, file, new_file, logger)

    ctx = _profile(label, file, new_file, logger, cleanup=torch.cuda.synchronize)
    return _ContextManager(
        action=ctx.__enter__, cleanup=lambda: ctx.__exit__(None, None, None)
    )  # pylint: disable=no-member


def profile_async(
    label: str,
    file: Union[str, os.PathLike, TextIO] = None,
    new_file: bool = False,
    logger: Optional[logging.Logger] = None,  # pylint: disable=redefined-outer-name
):
    """
    Profile a block of code and save profiling information into a file.

    :param label: String label associated with the block of code to profile (shows up in the profiling print)
    :param file: File path and name or a file-like object to send output text to (Default: stdout)
    :param new_file: True if a new file is to be created to hold profiling info, False if an existing file should be
        appended to. This flag is only valid when ``file`` is a path, not a file-like object.
    :param logger: If logger is provided, profiling string will also be printed with INFO logging level
    """
    ctx = _profile(label, file, new_file, logger, cleanup=None)
    return _ContextManager(
        action=ctx.__enter__, cleanup=lambda: ctx.__exit__(None, None, None)
    )  # pylint: disable=no-member


def is_vector_encoding(encoding: Optional[List[Dict]]) -> bool:
    """
    Check if encoding is from vector quantization

    :param encoding: List of encoding dictionaries
    :return: True if all required vector quantization properties are included in encoding
    """
    if encoding is None:
        return False

    required_properties = (
        "rows_per_block",
        "cols_per_block",
        "vector_dim",
        "vector_stride",
        "index_bw",
    )
    return all((property_ in encoding[0] for property_ in required_properties))


def get_all_named_parameters(model: torch.nn.Module):
    """
    Yields all (name, parameter) pairs in model including redundant parameters.

    :param model: torch.nn.Module from which to retrieve parameters
    """
    for name, module in model.named_modules(remove_duplicate=False):
        for param_name, parameter in module.named_parameters(recurse=False):
            if name:
                yield name + "." + param_name, parameter
            else:
                # Don't prepend . if module name is "" (Parameter owned by base model)
                yield param_name, parameter


@contextlib.contextmanager
def place_model(model: torch.nn.Module, device: torch.device):
    """
    Temporarily place model on given device
    """
    original_device = get_device(model)
    try:
        model.to(device=device)
        yield
    finally:
        model.to(device=original_device)


def get_param_channel_axis(module: torch.nn.Module, param_name: str):
    """
    Given a module and its param name, this method returns the channel axis of the given parameter.

    :param module: torch.nn.Module
    :param param_name: str representing the name of the parameter
    """
    channel_axis = 0
    if isinstance(
        module,
        (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d),
    ):
        channel_axis = 1 if param_name == "weight" else 0
    return channel_axis


def _is_expandable(src_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> bool:
    """
    Returns true if source shape can be expanded as target shape
    """
    if len(src_shape) > len(target_shape):
        return False

    for src_dim, dst_dim in zip(src_shape[::-1], target_shape[::-1]):
        if src_dim not in (1, dst_dim):
            return False

    return True


def _is_reducible(src_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> bool:
    """
    Returns true if source shape can be reduced as target shape
    """
    return _is_expandable(target_shape, src_shape)  # pylint: disable=arguments-out-of-order


def reduce(
    input: torch.Tensor,
    shape: tuple[int, ...],
    reduce_op: Callable,
    block_size: tuple[int, ...] | None = None,
):
    """
    Reduce input into given shape.

    :param input: Input to reduce
    :param shape: Shape of the reduced output
    :param reduce_op: Reduce operation
    :param block_size: Block size for block-wise reduction
    """
    output_shape = shape

    if block_size is not None:
        block_size = concretize_block_size(input.shape, shape, block_size)
        input = input.reshape(-1, *interleave(shape, block_size))
        shape = interleave(shape, 1)

    if not _is_reducible(input.shape, shape):
        raise RuntimeError(
            f"Input of shape {list(input.shape)} can't be reduced to shape {list(shape)}"
        )

    padded_shape = (*itertools.repeat(1, len(input.shape) - len(shape)), *shape)
    reduce_dims = tuple(axis for axis, dim in enumerate(padded_shape) if dim == 1)
    other_dims = tuple(axis for axis, dim in enumerate(padded_shape) if dim > 1)
    permute_dims = reduce_dims + other_dims

    return reduce_op(
        input.permute(permute_dims).reshape(-1, *output_shape), dim=0, keepdim=False
    )


class _ContextManager:
    def __init__(self, action: Callable[[], Any], cleanup: Callable[[], Any]):
        self._action = action
        self._cleanup = cleanup

    def __enter__(self):
        self._action()
        return self

    def __exit__(self, *_):
        self._cleanup()

    def __call__(self, fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)

        return wrapper


class _NullAttribute:
    pass


def patch_attr(obj, attr_name, new_attr) -> _ContextManager:
    """
    Temporarily overwrite object attribute
    """
    if isinstance(obj, torch.nn.Module):
        if attr_name in obj._parameters or attr_name in obj._buffers:  # pylint: disable=protected-access
            return _patch_param_or_buffer(obj, attr_name, new_attr)

    if hasattr(obj, attr_name):
        old_attr = getattr(obj, attr_name)
    else:
        old_attr = _NullAttribute()
    action = lambda: setattr(obj, attr_name, new_attr)

    def cleanup():
        try:
            delattr(obj, attr_name)
        except AttributeError:
            pass

        if not hasattr(obj, attr_name) and not isinstance(old_attr, _NullAttribute):
            setattr(obj, attr_name, old_attr)

    return _ContextManager(action, cleanup)


def _patch_param_or_buffer(
    module: torch.nn.Module,
    param_or_buffer_name: str,
    new_param_or_buffer: torch.Tensor,
):
    """
    Temporarily substitute the reference to the a parameter with the quantized parameter.
    Under the scope of this function, ``getattr(module, param_or_buffer_name)`` will return
    ``new_param_or_buffer`` instead of the original parameter.

    :param module: Module that owns the parameter
    :param param_or_buffer_name: Name of the parameter
    :param new_param_or_buffer: New parameter to replace the original parameter
    """
    # pylint: disable=protected-access

    orig_param_or_buffer = getattr(module, param_or_buffer_name)
    if orig_param_or_buffer is not None:
        assert new_param_or_buffer.shape == orig_param_or_buffer.shape

    if param_or_buffer_name in module._parameters:
        container = module._parameters
    elif param_or_buffer_name in module._buffers:
        container = module._buffers
    elif param_or_buffer_name in module.__dict__:
        # Some non-standard modules (e.g. replicas of torch.nn.DataParallel) store their parameters
        container = module.__dict__
    else:
        raise RuntimeError(
            f"'{param_or_buffer_name}' is not a valid name of parameter of buffer of {type(module)}."
        )

    action = lambda: container.update({param_or_buffer_name: new_param_or_buffer})
    cleanup = lambda: container.update({param_or_buffer_name: orig_param_or_buffer})

    return _ContextManager(action, cleanup)


class _StraightThroughEstimator(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, op, *args, **kwargs):  # pylint:disable=arguments-differ, unused-argument
        return op(*args, **kwargs)

    @staticmethod
    def backward(ctx, *grad):
        return (None, *grad)


def ste_round(*args, **kwargs):
    """
    Applies straight-through rounding
    """
    return _StraightThroughEstimator.apply(torch.round, *args, **kwargs)


class StatisticsNotFoundError(RuntimeError):
    """
    Error raised when compute_encodings() is invoked without statistics
    """


_ENABLE_RECOMPUTE = False


def _set_enable_recompute(mode: bool):
    original_mode = _ENABLE_RECOMPUTE

    def action():
        global _ENABLE_RECOMPUTE  # pylint: disable=global-statement
        _ENABLE_RECOMPUTE = mode

    def cleanup():
        global _ENABLE_RECOMPUTE  # pylint: disable=global-statement
        _ENABLE_RECOMPUTE = original_mode

    return _ContextManager(action, cleanup)


def is_recompute_enabled():
    """
    Returns True if recomputation for memory saving is enabled; False otherwise.
    """
    return _ENABLE_RECOMPUTE


def enable_recompute():
    """
    Enable recomputation for memory saving.
    """
    return _set_enable_recompute(True)


def no_recompute():
    """
    Disable recomputation for memory saving.
    """
    return _set_enable_recompute(False)


def allow_recompute(fn):
    """
    Allow recomputation of activation of the given function during training
    if recompute is enabled.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if is_recompute_enabled():
            # Enable activation recompute (a.k.a. activataion checkpointing)
            # to reduce memory footprint of training
            return torch.utils.checkpoint.checkpoint(
                fn, *args, use_reentrant=False, **kwargs
            )
        return fn(*args, **kwargs)

    return wrapper


def flatten_nn_module_list(module):
    """
    Flatten nested list of nn.Modules into a flat list
    """

    def flat_iter(mod):
        if isinstance(mod, (list, tuple, torch.nn.ModuleList)):
            for x in mod:
                yield from flat_iter(x)
        else:
            yield mod

    return list(flat_iter(module))


def _map_qmodule(modules, func):
    # pylint: disable=import-outside-toplevel
    # pylint: disable=protected-access, cyclic-import
    from aimet_torch.nn import BaseQuantizationMixin

    contexts = []
    ctx = _ContextManager(
        action=lambda: None,
        cleanup=lambda: [context._cleanup() for context in contexts],
    )

    if isinstance(modules, torch.nn.Module):
        modules = [modules]

    try:
        for module_elem in modules:
            for module in module_elem.modules():
                if isinstance(module, BaseQuantizationMixin):
                    context = func(module)
                    contexts.append(context)
    except Exception:
        ctx._cleanup()
        raise

    return ctx


def remove_input_quantizers(modules):
    """
    Temporarily remove all input quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_input_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): None
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
    """
    # pylint: disable=protected-access
    return _map_qmodule(modules, lambda qmodule: qmodule._remove_input_quantizers())


def remove_output_quantizers(modules):
    """
    Temporarily remove all output quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_output_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): None
            )
          )
        )
    """
    # pylint: disable=protected-access
    return _map_qmodule(modules, lambda qmodule: qmodule._remove_output_quantizers())


def remove_param_quantizers(modules):
    """
    Temporarily remove all parameter quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_param_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): None
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
    """
    # pylint: disable=protected-access
    return _map_qmodule(modules, lambda qmodule: qmodule._remove_param_quantizers())


def remove_activation_quantizers(modules):
    """
    Temporarily remove all input and output quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_activation_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): None
            )
            (output_quantizers): ModuleList(
              (0): None
            )
          )
        )
    """
    if not isinstance(modules, torch.nn.Module):
        # Shallow copy in case modules is an iterator
        modules = list(modules)

    context_1 = remove_input_quantizers(modules)
    context_2 = remove_output_quantizers(modules)
    # pylint: disable=protected-access
    return _ContextManager(
        action=lambda: None,
        cleanup=lambda: (context_1._cleanup(), context_2._cleanup()),
    )


def remove_all_quantizers(modules):
    """
    Temporarily remove all quantizers

    Example:

        >>> print(sim.model)
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): QuantizeDequantize(shape=(3, 1, 1, 1), qmin=-128, qmax=127, symmetric=True)
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
            (output_quantizers): ModuleList(
              (0): QuantizeDequantize(shape=(), qmin=0, qmax=255, symmetric=False)
            )
          )
        )
        >>> with remove_all_quantizers(sim.model):
        ...     print(sim.model)
        ...
        Sequential(
          (0): QuantizedConv2d(
            3, 3, kernel_size=(3, 3), stride=(1, 1)
            (param_quantizers): ModuleDict(
              (weight): None
              (bias): None
            )
            (input_quantizers): ModuleList(
              (0): None
            )
            (output_quantizers): ModuleList(
              (0): None
            )
          )
        )
    """
    if not isinstance(modules, torch.nn.Module):
        # Shallow copy in case modules is an iterator
        modules = list(modules)

    context_1 = remove_activation_quantizers(modules)
    context_2 = remove_param_quantizers(modules)
    # pylint: disable=protected-access
    return _ContextManager(
        action=lambda: None,
        cleanup=lambda: (context_1._cleanup(), context_2._cleanup()),
    )


def has_no_quantizers(module, ignore_params: bool = False) -> bool:
    """
    Helper function to check if a module has any quantizers enabled
    """
    return (
        all(inp_qtzr is None for inp_qtzr in module.input_quantizers)
        and all(out_qtzr is None for out_qtzr in module.output_quantizers)
        and (
            ignore_params
            or all(
                param_qtzr is None for param_qtzr in module.param_quantizers.values()
            )
        )
    )


def rgetattr(obj, attr):
    """Drop in replacement for __getattr__ that can handle dotted attribute strings"""
    return functools.reduce(getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    """Drop in replacement for __setattr__ that can handle dotted attribute strings"""
    pre, _, post = attr.rpartition(".")
    pre_obj = rgetattr(obj, pre) if pre else obj
    return setattr(pre_obj, post, val)


def apply_fn_recursively_to_all_elems(fn, container):
    """Apply fn to all elements in recursively composed container"""
    if container is None:
        return None
    if isinstance(container, (list, tuple)):
        return [apply_fn_recursively_to_all_elems(fn, elem) for elem in container]
    if isinstance(container, dict):
        return {
            key: apply_fn_recursively_to_all_elems(fn, elem)
            for key, elem in container.items()
        }
    return fn(container)


def flatten_list(container):
    """Helper function to flatten nested list/tuple into 1D"""
    if not container:
        return container
    if not isinstance(container, (list, tuple)):
        return [container]
    if isinstance(container[0], (list, tuple)):
        return flatten_list(container[0]) + flatten_list(container[1:])
    if len(container) == 1:
        return container
    return container[:1] + flatten_list(container[1:])


def default_forward_fn(model, inputs):
    """
    Default forward function.
    :param model: pytorch model
    :param inputs: model inputs
    """
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
    return model(*inputs)


_torch_compiler_is_compiling: Callable[[], bool]
_torch_compiler_is_dynamo_compiling: Callable[[], bool]
_torch_compiler_is_exporting: Callable[[], bool]

if version.parse(torch.__version__) >= version.parse("2.7"):
    _torch_compiler_is_compiling = torch.compiler.is_compiling
    _torch_compiler_is_dynamo_compiling = torch.compiler.is_dynamo_compiling
    _torch_compiler_is_exporting = torch.compiler.is_exporting
else:
    # torch < 2.7.0 doesn't have torch.compiler.is_compiling/exporting API
    def _torch_compiler_is_compiling() -> bool:
        return False

    def _torch_compiler_is_dynamo_compiling() -> bool:
        return False

    def _torch_compiler_is_exporting() -> bool:
        return False


_qtensor_enabled = True


@contextmanager
def _enable_qtensor_casting(enable: bool):
    global _qtensor_enabled  # pylint: disable=global-statement

    original_value = _qtensor_enabled
    try:
        _qtensor_enabled = enable
        yield
    finally:
        _qtensor_enabled = original_value


def _is_qtensor_casting_enabled():
    return _qtensor_enabled


@contextmanager
def _inference_mode(model: torch.nn.Module, prequantize_parameters: bool):
    # pylint: disable=protected-access
    from .quantsim import QuantizationSimModel
    from .quantization.tensor import DequantizedTensor
    from .quantization.base import QuantizerBase

    with ExitStack() as stack:
        for q in model.modules():
            if isinstance(q, QuantizerBase):
                dtype = next(
                    p.dtype for p in itertools.chain(q.parameters(), q.buffers())
                )
                stack.enter_context(q._precompute_encodings(dtype=dtype))

        stack.enter_context(_enable_qtensor_casting(False))

        if prequantize_parameters:
            stack.enter_context(
                QuantizationSimModel._apply_qdq_to_model_parameters(model)
            )
            stack.enter_context(remove_param_quantizers(model))

        def cast_param_to_plain_tensor(module):
            for name, param in module.named_parameters(recurse=False):
                if isinstance(param, DequantizedTensor):
                    stack.enter_context(
                        patch_attr(module, name, param.as_subclass(torch.Tensor))
                    )

        model.apply(cast_param_to_plain_tensor)
        yield


class _DecompositionError(RuntimeError):
    pass


@torch.no_grad()
def _decompose_2bit_prequantized_tensor(
    input_qdq: torch.Tensor,
    scale_shape: tuple[int, ...],
    block_size: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a pre-quantized tensor (input_qdq)
    into an integer tensor (input_q) and float scale (scale)

    scale := gcd(input_qdq)
    input_q := input_qdq / scale
    """
    from aimet_torch.quantization.affine import quantize

    if len(scale_shape) > len(input_qdq.shape):
        raise ValueError

    concrete_block_size = concretize_block_size(
        input_qdq.shape, scale_shape, block_size or tuple(-1 for _ in scale_shape)
    )
    concrete_block_size = (
        *input_qdq.shape[: len(input_qdq.shape) - len(concrete_block_size)],
        *concrete_block_size,
    )
    input_qdq_abs = input_qdq.abs()
    input_qdq_abs = input_qdq_abs.reshape(
        *interleave(
            [dim // b for dim, b in zip(input_qdq.shape, concrete_block_size)],
            concrete_block_size,
        )
    )
    reduce_dims = tuple(range(1, input_qdq_abs.ndim, 2))
    absmax = input_qdq_abs.amax(dim=reduce_dims, keepdim=True)

    if torch.all((input_qdq_abs == 0) | (input_qdq_abs == absmax)):
        minimum_scale = _get_minimum_scale(2**2 - 1)
        scale = absmax.reshape(scale_shape)
        scale.masked_fill_(scale == 0, minimum_scale)
        input_q = quantize(
            input_qdq,
            scale,
            offset=torch.zeros_like(scale),
            qmin=-1,
            qmax=1,
            block_size=block_size,
        )
        return input_q, scale

    raise _DecompositionError(
        "Failed to decompose input tensor into input_q (int2) and scale (float)"
    )
