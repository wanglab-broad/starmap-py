from importlib.metadata import version

from starmap import tools as tl
# from starmap import preprocessing as pp
# from starmap import plotting as pl
# from starmap import archive as ar

import sys

# sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['tl', 'pp', 'pl', 'ar']})
# from ._utils import annotate_doc_types

# annotate_doc_types(sys.modules[__name__], 'starmap')
# del sys, annotate_doc_types

# __all__ = ["pl", "pp", "tl", "ar"]

__version__ = version("starmap-py")
