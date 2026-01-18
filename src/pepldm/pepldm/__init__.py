from typing import TYPE_CHECKING

from transformers.utils import _LazyModule
from transformers.utils.import_utils import define_import_structure

define_import_structure = {
    "configuration_pepldm": ["PepLDMConfig"],
    "modeling_pepldm": ["PepLDMModel"],
}

if TYPE_CHECKING:
    from .configuration_pepldm import *
    from .modeling_pepldm import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(
        __name__, _file, define_import_structure, module_spec=__spec__
    )
