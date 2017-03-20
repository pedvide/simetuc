# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:16:47 2017

@author: Villanueva
"""


import sys
#import sys
import logging
# nice debug printing of settings
#import pprint
from typing import Dict, List, Tuple
from typing import Callable, TypeVar
from typing import Union, Sequence, Iterable, Mapping, Collection  # type: ignore
from functools import wraps
from enum import Enum
import warnings

from simetuc.util import ConfigWarning


## example on how to add a type to the typing module, in this case a OrderedDict
#import collections
#import typing
#class OrderedDictType(collections.OrderedDict, typing.MutableMapping[typing.KT, typing.VT],
#                      extra=collections.OrderedDict):
#    __slots__ = ()
#    def __new__(cls, *args, **kwds):
#        if typing._geqv(cls, OrderedDictType):
#            raise TypeError("Type OrderedDictType cannot be instantiated; "
#                            "use collections.OrderedDict() instead")
#        return typing._generic_new(collections.OrderedDict, cls, *args, **kwds)
#typing.OrderedDictType = OrderedDictType
#typing.OrderedDictType.__module__ = 'typing'


# type of innermost type, could be int, str, etc
T = TypeVar('T')
# the type of the settings value can be a simple type or a collection or types
ValType = Union[T, Collection[T]]

class Value():
    '''A value of a setting. The value has an specific type and optionally max and min values.
        If the setting is a list, the list can have max and min length.
        The type can be nested, eg: a list of lists of ints; use the typing module for that:
        type=typing.List[typing.List[int]].
        If the setting is a nested sequence, the max and min length
        can be lists of the length of each list.
        At any level of the type tree a Union of types is allowed; if present, the leftmost type
        in the Union will be parsed, if it doesn't succeed the next type will be parsed and so on.
        The innermost type must be concrete or a typing.Union of several concrete types.
        In this case the type of the settings will be checked agains in the values in the Union
        (in the same order) until one matches, or if none, an exception is raised.'''

    class Kind(Enum):
        mandatory = 1
        optional = 2
        exclusive = 3

    mandatory = Kind.mandatory
    optional = Kind.optional
    exclusive = Kind.exclusive

    # len_max/min are ints b/c the length of a Sequence is always an int.
    def __init__(self, name: str, val_type: ValType,
                 val_max: T = None, val_min: T = None,
                 kind: Kind = Kind.mandatory,
                 len_max: Union[int, List[int]] = None,
                 len_min: Union[int, List[int]] = None) -> None:
        '''Val type can be a nested type (List[int], List[List[int]])'''

        self.name = name
        # val_type it's either a simple type, an Iterable or a Union
        # Nesting is possible
        self.val_type = val_type

        self.val_max = val_max
        self.val_min = val_min

        self.kind = kind

        # convert to list
        self.len_max = [len_max] if not isinstance(len_max, Sequence) else len_max
        self.len_min = [len_min] if not isinstance(len_min, Sequence) else len_min

        self.no_logging = False

    def __repr__(self) -> str:
        '''Return a representation of Value'''

        return '{}({}, {})'.format(self.__class__.__name__, self.name,
                                   _get_clean_type_name(self.val_type))

    @staticmethod
    def _print_trace(value: ValType, val_type: ValType) -> None:  # pragma: no cover
        '''Print the current state of the tree parsing.'''
        type_name = _get_clean_type_name(val_type)
        print(value, type(value).__name__, type_name)

    def _check_val_max_min(self, value: T) -> None:
        '''Check that the value is within the max and min values.'''
        logger = logging.getLogger(__name__)

        # the type: ignore comments are there because it's possible that the user will
        # ask for an unorderable type but also give a max or min values.
        # That would fail, but that's the user's fault.
        if self.val_max is not None and value > self.val_max:  # type: ignore
            msg = ('Value(s) of {} ({}) cannot be '.format(self.name, value) +
                   'larger than {}.'.format(self.val_max))
            logger.error(msg)
            raise ValueError(msg)
        if self.val_min is not None and value < self.val_min:  # type: ignore
            msg = ('Value(s) of {} ({}) cannot be '.format(self.name, value) +
                   'smaller than {}.'.format(self.val_min))
            logger.error(msg)
            raise ValueError(msg)

    def _check_seq_len(self, seq: Collection, len_max: int, len_min: int) -> None:
        '''Checks that the sequence has the size given by self.len_max/min.'''
        logger = logging.getLogger(__name__)

        if len_max is not None and len(seq) > len_max:
            msg = ('Length of {} ({}) cannot be '.format(self.name, len(seq)) +
                   'larger than {}.'.format(len_max))
            logger.error(msg)
            raise ValueError(msg)
        if len_min is not None and len(seq) < len_min:
            msg = ('Length of {} ({}) cannot be '.format(self.name, len(seq)) +
                   'smaller than {}.'.format(len_min))
            logger.error(msg)
            raise ValueError(msg)

    def _raise_wrong_type_error(self, value: ValType, val_type: ValType, err: Exception = None) -> None:
        '''Raises ValueError because the type of the value is not the expected val_type.'''
        logger = logging.getLogger(__name__)

        msg1 = 'Setting {!r} (value: {!r}, type: {}) does not '.format(self.name, value,
                                                                       type(value).__name__)
        msg2 = 'have the right type ({}).'.format(_get_clean_type_name(val_type))

        err_str = ' ' + str(err) if err is not None else ''

        if self.no_logging is False:
            logger.error(msg1 + msg2 + err_str)
        raise ValueError(msg1 + msg2 + err_str)

    def _cast_to_type(self, value: ValType, val_type: ValType) -> ValType:
        '''Cast the value to the type, which should be callable.'''
        try:
            parsed_value = val_type(value)
        except (ValueError, TypeError) as err:  # no match
            self._raise_wrong_type_error(value, val_type, err)
        else:
            # no exception, val_type matched the value!
            return parsed_value

    @staticmethod
    def trace(fn: Callable) -> Callable:  # pragma: no cover
        '''Trace the execution of a recursive function'''
        stream = sys.stdout
        indent_step = 2
        show_ret = False
        cur_indent = 0
        @wraps(fn)
        def wrapper(*args: Tuple, **kwargs: Dict) -> None:
            nonlocal cur_indent
            indent = ' ' * cur_indent
            argstr = ', '.join(
                [repr(a).replace('typing.', '') for a in args[1:3]])
            stream.write('%s%s(%s)\n' % (indent, fn.__name__, argstr))

            cur_indent += indent_step
            ret = fn(*args, **kwargs)
            cur_indent -= indent_step

            if show_ret:
                stream.write('%s--> %s\n' % (indent, ret))
            return ret
        return wrapper

#    @trace
    def _parse_type_tree(self, value: ValType, val_type: ValType,
                         len_max: List = None, len_min: List = None,
                         key: bool = False) -> ValType:
        '''Makes sure that the sequence/value has recursively the given type, ie:
            a = [1, 2, 3] has val_type=List, and then val_type=int
            b = [[1, 2, 3], [4, 5, 6]] has val_type=List, then val_type=List and val_type=int
            c = 1 has val_type=int.
            The innermost type has to be a concrete one (eg: int, str, etc) or an Union of
            concrete types; in this case each type in the Union will be tested in order.
            Returns the parsed sequence/value.
            len_max/min is a list with the max and min list length at this and lower
            tree levels. The values can be None
        '''
        # Typing module types have an __extra__ attribute with the actual instantiable type,
        # ie: Tuple.__extra__ = tuple
        # Sequence types also have an __args__ attribute with a tuple of the inner type(s),
        # ie: List[int].__args__ = (int,)
        # Union types also have and __args__ attribute with the types of the union.
        # Same with Mappings

        logger = logging.getLogger(__name__)
#        Value._print_trace(value, val_type)
#        print(value, val_type)

        # length max and min at this tree level
        cur_len_max = None if not len_max else len_max[0]
        cur_len_min = None if not len_min else len_min[0]
        rest_len_max = [None] if len(len_max) < 1 else len_max[1:]
        rest_len_min = [None] if len(len_min) < 1 else len_min[1:]

        # Union type, try parsing each option until one works
        if _is_union(val_type):
            type_list = val_type.__args__
            for curr_type in type_list:
                try:
                    self.no_logging = True
                    parsed_value = self._parse_type_tree(value, curr_type,
                                                         len_max, len_min)
                except ValueError as err:
                    # save exception and traceback for later
                    last_err = err
                    tb = sys.exc_info()[2]
                    continue
                else:
                    # some type in the Union matched
                    return parsed_value
                finally:
                    self.no_logging = False
            # no match, error
            msg = ('Setting {!r} (value: {!r}, type: {}) does not have '
                   'any of the right types ({})')
            msg = msg.format(self.name, value, type(value).__name__,
                             ', '.join(_get_clean_type_name(typ)
                                       for typ in val_type.__args__))
            logger.error(msg)
            logger.error(str(last_err))
            raise ValueError(msg + ', because ' + str(last_err)).with_traceback(tb)

        # ValueDicts
        elif isinstance(val_type, DictValue):
            if not isinstance(value, Mapping):
                self._raise_wrong_type_error(value, val_type)

            # parse all values in the dictionary
            parsed_dict = self._cast_to_type(value, val_type)

            return parsed_dict

        # generic mappings such as Dicts: parse both the keys and the values
        elif issubclass(val_type, Mapping) and hasattr(val_type, '__extra__'):
            if not isinstance(value, Mapping):
                self._raise_wrong_type_error(value, val_type)
            # go through all keys and values and parse them
            # __args__ has the two types for the keys and values
            mapping = {self._parse_type_tree(inner_key, val_type.__args__[0],
                                             rest_len_max, rest_len_min, key=True):
                       self._parse_type_tree(inner_val, val_type.__args__[1],
                                             rest_len_max, rest_len_min)
                       for inner_key, inner_val in value.items()}

            # check length
            self._check_seq_len(mapping, cur_len_max, cur_len_min)

            return self._cast_to_type(mapping, val_type.__extra__)

        # single concrete type (int, str, list, dict, ...): cast to correct type
        elif not hasattr(val_type, '__extra__'):
            parsed_value = self._cast_to_type(value, val_type)
            # don´t check if key is True: it's a mapping key
            if not key:
                self._check_val_max_min(parsed_value)
            return parsed_value

        # generic sequences such as Lists, Tuples: parse each item
        else:
            # first check that lst is of the right type
            # str behave like lists, so if the user wanted a list and value is a str,
            # cast_to_type will succeed! So avoid it,
            # also avoid iterating if value is not iterable
            if (isinstance(value, str) and not issubclass(val_type, str)
                    or not isinstance(value, Iterable)):
                self._raise_wrong_type_error(value, val_type)

            if val_type.__args__ is None:
                msg = 'Invalid requested type ({}), generic types must contain arguments.'
                msg = msg.format(_get_clean_type_name(val_type))
                logger.error(msg)
                raise ValueError(msg)

            # build sequence from the lower branches,
            # pass the lower level lengths
            sequence = [self._parse_type_tree(inner_lst, val_type.__args__[0],
                                              rest_len_max, rest_len_min)
                        for inner_lst in value]

            # check length
            self._check_seq_len(sequence, cur_len_max, cur_len_min)

            return self._cast_to_type(sequence, val_type.__extra__)

    def parse(self, value: ValType) -> ValType:
        '''Parses the value from a settings file
            and tries to convert it to this Value's type.'''
        return self._parse_type_tree(value, self.val_type, self.len_max, self.len_min)


def _is_union(val_type: ValType) -> bool:
    '''Is val_type an Union?'''
    return type(val_type) == type(Union)  # pylint: disable=C0123


def _get_clean_type_name(val_type: ValType) -> str:
    '''Returns the clean name of the val_type'''
    if val_type.__module__ == 'typing':
        type_name = str(val_type).replace('typing.', '')
    else:
        type_name = val_type.__name__
    return type_name.replace('__main__.', '')



class DictValue():
    '''Represents a dictionary of Values, each with a name and type.'''
    mandatory = Value.mandatory
    optional = Value.optional
    exclusive = Value.exclusive

    def __init__(self, name: str, values_list: List[ValType],
                 kind: Value.Kind = Value.mandatory) -> None:
        self.name = name
        self.values_list = values_list
        self.kind = kind
        self.__name__ = '{}({})'.format(self.__class__.__name__,
                                   ', '.join(repr(value).replace('Value', '')
                                             for value in self.values_list))

    def __repr__(self) -> str:
        return self.__name__

    def __call__(self, config_dict: Dict) -> Dict:
        return self.parse(config_dict)

    def parse(self, config_dict: Dict) -> Dict:
        '''Return the parsed dictionary'''
        logger = logging.getLogger(__name__)

        self._check_extra_and_exclusive(config_dict)

        #  we are given a dictionary to match
        parsed_dict = {}  # store parsed values
        for val in self.values_list:
            try:
#                print(val)
                # skip optional values that aren't present
                if val.kind is not Value.mandatory and val.name not in config_dict:
                    continue
                parsed_dict.update({val.name: val.parse(config_dict[val.name])})
            except (KeyError, TypeError) as err:
                msg = 'Mandatory value "{}" not found in section "{}".'.format(val.name, self.name)
                logger.error(msg)
                raise ValueError(msg) from err

        return parsed_dict

    def _check_extra_and_exclusive(self, config_dict: Dict) -> None:
        '''Check that exclusive values are not present at the same time'''
        logger = logging.getLogger(__name__)
        try:
            present_values = set(config_dict.keys())
        except (AttributeError, TypeError) as err:
            msg = 'Section "{}" is empty or otherwise invalid!'.format(self.name)
            logger.error(msg, exc_info=True)
            raise ValueError(msg) from err

        needed_values = set(val.name for val in self.values_list if val.kind is Value.mandatory)
        optional_values = set(val.name for val in self.values_list if val.kind is Value.optional)
        exclusive_values = set(val.name for val in self.values_list if val.kind is Value.exclusive)
        optional_values = optional_values | exclusive_values

        set_extra = present_values - needed_values
        # if there are extra values and they aren't optional
        if set_extra and not set_extra.issubset(optional_values):
            set_not_optional = set_extra - optional_values
            logger.warning('WARNING! The following values in section "%s" ' +
                           'are not recognized:', self.name)
            logger.warning(str(set_not_optional))
            warnings.warn('Some values or sections should not be present in the file.',
                          ConfigWarning)

        # exclusive values
        if exclusive_values and exclusive_values.issubset(present_values):
            logger.error('The following values in section "%s"'  +
                         ' are mutually exclusive: ', self.name)
            logger.error(str(exclusive_values))
            raise ValueError('Only one of the values {} must be present.'.format(exclusive_values))

