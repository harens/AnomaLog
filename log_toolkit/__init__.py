import logging

from beartype.claw import beartype_this_package

beartype_this_package()

logging.getLogger(__name__).addHandler(logging.NullHandler())
