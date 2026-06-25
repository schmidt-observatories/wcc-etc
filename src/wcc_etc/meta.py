import warnings
from copy import deepcopy


class _MetaHolder_:
    """
    A base class to handle metadata and mutable parameters.

    Attributes
    ----------
    meta : dict
        Current metadata/parameters.
    mutable_parameters : list
        List of parameters that are allowed to be updated.
    """

    _mutable_parameters = []

    def __init__(self, meta={}):
        """
        Initialize the _MetaHolder_.

        Parameters
        ----------
        meta : dict, optional
            Initial metadata dictionary. Default is {}.
        """
        self._meta = deepcopy(meta)  # do not affect input dict.
        self._meta_in = deepcopy(self._meta)

    # ============== #
    #  Methods       #
    # ============== #
    def reset(self):
        """
        Revert the metadata to the initial input parameters.
        """
        self._meta = deepcopy(self._meta_in)

    def update(self, reset=False, **kwargs):
        """
        Change any mutable parameter.

        Parameters
        ----------
        reset : bool, optional
            Whether to reset parameters to their initial configuration before updating.
            Default is False.
        **kwargs
            Parameters to be updated.

        Returns
        -------
        dict_keys
            The keys of the parameters that have been updated.
        """
        updated_params = {}
        for key, value in kwargs.items():
            # this trick enables to pass update(a=None) while doing nothing on "a"
            if value is None:
                continue

            if key not in self.mutable_parameters:
                warnings.warn(f"{key=} is not a mutable parameter. *ignored*")
                continue

            # looks good, let's udpate that.
            updated_params[key] = value

        # update the parameters. Should it be reset ?
        if reset:
            self.reset()

        self._meta |= updated_params
        return updated_params.keys()

    def describe(self):
        """
        Print the current parameters and their values.
        """
        for key, value in self.meta.items():
            print(f"  {key}: {value}")

    # ================ #
    #   Properties     #
    # ================ #
    @property
    def meta(self):
        """
        The current metadata dictionary.
        """
        return self._meta

    @property
    def mutable_parameters(self):
        """
        The list of parameters that are allowed to be updated.
        """
        return self._mutable_parameters
