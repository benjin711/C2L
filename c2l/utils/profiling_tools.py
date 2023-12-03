import logging
import timeit
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FlameGraphOptions(Enum):
    """
    Visualization options for the flame graph.
    """
    SELF_CPU_TIME_TOTAL = "self_cpu_time_total"
    SELF_CUDA_TIME_TOTAL = "self_cuda_time_total"


@dataclass
class PytorchProfileConfig:
    """
    Configuration for the pytorch profiler.
    """
    activities: List[ProfilerActivity] = field(
        default_factory=lambda: [ProfilerActivity.CPU, ProfilerActivity.CUDA])
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = False
    with_modules: bool = False

    def asdict(self):
        return asdict(self)

    def __repr__(self):
        return str(asdict(self))


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """
    Count number of parameters and buffer parameters in a model.
    Args:
        model: PyTorch model.
        only_trainable: If True, count only trainable parameters.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = sum(p.numel() for p in model.parameters())
    num_buffers = sum(p.numel() for p in model.buffers())

    return num_params + num_buffers


def calc_memory_footprint(model: nn.Module) -> int:
    """
    Calculate the memory footprint of a model.
    Args:
        model: PyTorch model.
    """
    mem_params = sum(param.numel() * param.element_size() for param in model.parameters())
    mem_bufs = sum(buf.numel() * buf.element_size() for buf in model.buffers())

    return mem_params + mem_bufs  # in bytes


def dump_device_memory_history(
    func: Callable,
    device: str = "cuda",
    output_file: str = "snapshot.pickle"
) -> None:
    """
    Dump memory history of a device. The resulting pickle file can be visualized as described here:
    https://pytorch.org/docs/stable/torch_cuda_memory.html#torch-cuda-memory
    Args:
        func: Function to profile.
        device: Device to profile.
        output_file: Output file for the memory history.
    """
    # pylint: disable=protected-access
    torch.cuda.memory._record_memory_history(device=device)

    # output is needed to make sure the final memory footprint is correct
    output = func()  # pylint: disable=unused-variable

    # pylint: disable=protected-access
    torch.cuda.memory._dump_snapshot(output_file)
    logger.info(f"Dumped memory history to {output_file}")


def memory_profile_model(model: nn.Module, **inputs: Any) -> Dict:
    """
    Profile the memory consumption of a model.
    Args:
        model: PyTorch model.
        inputs: Inputs to the model.
    Returns:
        Dictionary with memory consumption statistics.
    """
    stats = {}

    stats['num_params'] = count_parameters(model)
    stats['model_memory_footprint'] = calc_memory_footprint(model)

    stats['input_memory_footprint'] = 0
    for v in inputs.values():
        stats['input_memory_footprint'] += v.element_size() * v.numel()

    # output is needed to make sure the final memory footprint is correct
    output = model(**inputs)  # pylint: disable=unused-variable

    stats['peak_memory_forward'] = torch.cuda.max_memory_allocated()
    stats['final_memory_forward'] = torch.cuda.memory_allocated()

    return stats


def profile_runtime_flame_graph(
    func: Callable,
    profile_config: PytorchProfileConfig,
    flame_graph_option: FlameGraphOptions,
    output_file: str = "profiler_stacks.txt",
) -> None:
    """
    Profile the runtime of a function using the pytorch profiler
    and visualize the results as a flame graph. For more information:
    https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    Args:
        func: Function to profile.
        profile_config: Configuration for the profiler.
        flame_graph_option: Visualization option for the flame graph.
        output_file: Output file for the flame graph.
    """
    # with_stack = True required for flame graph
    profile_config.with_stack = True

    # Warmup
    func()

    # pylint: disable=protected-access
    with profile(**(profile_config.asdict()),
                 experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        with record_function("function"):
            func()

    prof.export_stacks(output_file, flame_graph_option.value)
    logger.info(f"Exported flame graph to {output_file}")


def profile_runtime_stack_traces(
    func: Callable,
    profile_config: PytorchProfileConfig,
    output_file: str = "trace.json",
) -> None:
    """
    Profile the runtime of a function using the pytorch profiler
    and visualize the results as a stack trace. For more information:
    https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    Args:
        func: Function to profile.
        *args: Arguments to pass to the function.
        output_file: Output file for the json stack traces.
        profile_config: Configuration for the profiler.
    """
    # Warmup
    func()

    with profile(**(profile_config.asdict())) as prof:
        with record_function("function"):
            func()

    prof.export_chrome_trace(output_file)
    logger.info(f"Exported stack traces to {output_file}")


def profile_dataloading_bottleneck(dataloader: DataLoader) -> None:
    """
    Profile dataloading bottleneck using an empty loop.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader to profile
    """
    def _profile_dataloading():
        # pylint: disable=unused-variable
        for batch in tqdm(dataloader, desc="Profiling dataloading"):
            pass

    t = timeit.timeit(_profile_dataloading, number=1)
    logger.info("Finished profiling dataloading")

    num_batches = len(dataloader)
    logger.info(f"Number of batches: {num_batches}")
    logger.info(f"Total time: {t:.2f}s")
    logger.info(f"Time per batch: {t / num_batches:.2f}s")
