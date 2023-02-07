from importlib import import_module

def load_initializer(initializer_cls_string):
    """
    Loads an initializer class from a string.
    """
    initializer_module_name, initializer_class_name = initializer_cls_string.rsplit('.', 1)
    initializer_module = import_module(initializer_module_name)
    initializer_cls = getattr(initializer_module, initializer_class_name)
    return initializer_cls

def initialize_env(env, initializer, initial_obs):
    """
    Fully initializes an environment using an initializer

    Args:
        env: the environment to initialize
        initializer: the initializer to use
        initial_obs: the initial observation
    """
    initializer.reset()
    action, done = initializer.get_action(initial_obs)
    obs, _, env_done, info = env.step(action)

    while not done:
        if env_done:
            raise RuntimeError("Environment is done before initializer is done!")

        action, done = initializer.get_action(obs, info)
        obs, _, env_done, info = env.step(action)
    
    return obs, env_done, info
