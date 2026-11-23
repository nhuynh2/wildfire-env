__all__ = ["DroneWildfireEnv", "WildfireEnv"]

def __getattr__(name):
    if name == "DroneWildfireEnv":
        from .drone_env import DroneWildfireEnv
        return DroneWildfireEnv
    if name == "WildfireEnv":
        from .wildfire_env import WildfireEnv
        return WildfireEnv
    raise AttributeError(name)
