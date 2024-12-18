class ml_config():
    def __init__(self, config_module):
        for attr in dir(config_module):
            # Skip special/private attributes
            if not attr.startswith("__"):
                setattr(self, attr, getattr(config_module, attr))
