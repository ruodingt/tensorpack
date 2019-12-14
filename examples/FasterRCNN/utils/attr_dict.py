import yaml
import pprint


class AttrDict(object):
    # def __init__(self):
    _frozen = False  # If True: Avoid accidental creation of new hierarchies.
    # _deep_frozen = False  # If True: avoid any modification

    def __getattr__(self, name):
        if self._frozen:
            raise AttributeError("You can't access strange stuff ({}) under frozen mode".format(name))
        if name.startswith('_'):
            # Do not mess with internals. Otherwise copy/pickle will fail
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __setattr__(self, name, value):
        if self._frozen and name != '_frozen':
            raise AttributeError('frozen config \"{}\" is not allowed to be modified'.format(name))
        # if self._frozen and name not in self.__dict__ and name != '_deep_frozen':
        #     raise AttributeError(
        #         "Config was frozen! Unknown config: {}".format(name))
        super().__setattr__(name, value)

    def __str__(self):
        return pprint.pformat(self.to_dict(), indent=1, width=100, compact=True)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__.items() if not k.startswith('_')}

    def from_dict(self, d):
        self.freeze(False)
        for k, v in d.items():
            self_v = getattr(self, k)
            if isinstance(self_v, AttrDict):
                self_v.from_dict(v)
            else:
                setattr(self, k, v)

    def update_config_from_args(self, args):
        """Update from command line args. """
        for cfg in args:
            keys, v = cfg.split('=', maxsplit=1)
            keylist = keys.split('.')

            dic = self
            for i, k in enumerate(keylist[:-1]):
                assert k in dir(dic), "Unknown config key: {}".format(keys)
                dic = getattr(dic, k)
            key = keylist[-1]

            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                v = eval(v)
            setattr(dic, key, v)

    def freeze(self, frozen=True):
        self._frozen = frozen
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.freeze(frozen)

    # avoid silent bugs
    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()

    def to_yaml(self, output_path):
        d = self.to_dict()
        if output_path:
            with open(output_path, 'w') as outfile:
                yaml.dump(d, outfile, default_flow_style=False)
            print('Config written to {}...'.format(output_path))
        else:
            return yaml.dump(d, None, default_flow_style=False)
