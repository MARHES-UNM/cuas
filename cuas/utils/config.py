class Config(dict):
    """Simple method to turn dictionary into attributes for easy access
    https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    https://stackoverflow.com/questions/3031219/recursively-access-dict-via-attributes-as-well-as-index-access
    """

    MARKER = object()

    def __new__(cls, value):
        """Makes the class a singleton"""
        if not hasattr(cls, "instance"):
            cls.instance = super(Config, cls).__new__(cls, value)
        return cls.instance

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError("Expected dict")

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super(Config, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, Config.MARKER)
        if found is Config.MARKER:
            found = Config()
            super(Config, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__