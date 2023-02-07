from abc import ABC, abstractmethod

class InitializerBase(ABC):
    """
    Base class for initializers.
    """
    @abstractmethod
    def __init__(self, env, args):
        """
        Initialize the initializer. Variant `args` are passed in for convenience.

        Args:
            env: Environment object
            args: Argument class
        """
        pass

    def reset(self):
        """
        (Optional) Called when environment is reset.
        Useful for stateful initializers that need to re-init
        themselves when a new environment is created
        """
        pass

    @abstractmethod
    def get_action(self, obs, info=None):
        """
        Get action from the initializer.

        NOTE: `info` is NOT AVAILABLE on the first environment step!
        If `info` is always needed, just sample something arbitrary
        on the first step. Can also use `env` information saved from
        __init__.
        
        However, good practice is to only use information from `obs`, as
        this is what the initializer will have access to in real.

        Args:
            obs: observation from the environment
            info: info from the environment

        Returns:
            action: action to be taken
            done: True if initializer is complete
        """
        pass
