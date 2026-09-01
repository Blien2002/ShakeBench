from robosuite.wrappers.wrapper import Wrapper
from robosuite.wrappers.data_collection_wrapper import DataCollectionWrapper
from robosuite.wrappers.demo_sampler_wrapper import DemoSamplerWrapper
from robosuite.wrappers.domain_randomization_wrapper import DomainRandomizationWrapper
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper

try:
    from robosuite.wrappers.gym_wrapper import GymWrapper
except ImportError as _gym_wrapper_import_error:
    # Keep the other robosuite wrappers importable without the optional Gym
    # dependency, while making an explicit GymWrapper import fail clearly.
    _GYM_WRAPPER_IMPORT_ERROR = _gym_wrapper_import_error

    def __getattr__(name):
        if name == "GymWrapper":
            message = str(_GYM_WRAPPER_IMPORT_ERROR)
            raise ImportError(message) from _GYM_WRAPPER_IMPORT_ERROR
        raise AttributeError(name)
