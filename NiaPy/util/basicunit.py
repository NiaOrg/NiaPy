# encoding=utf8

r"""Basic Units."""

from six import with_metaclass

import numpy as np

import matplotlib.units as units
import matplotlib.ticker as ticker

class ProxyDelegate:
    r"""Proxy delegate."""
    def __init__(self, fn_name, proxy_type):
        r"""Construct proxy delegate.

        Args:
            fn_name (str): TODO.
            proxy_type (Any): TODO.
        """
        self.proxy_type = proxy_type
        self.fn_name = fn_name

    def __get__(self, obj, objtype=None):
        r"""Get object.

        Args:
            obj (Any): Object.
            objtype (Optional[Any]): Type of the object.

        Returns:
            Any: Object.
        """
        return self.proxy_type(self.fn_name, obj)


class TaggedValueMeta(type):
    r"""Tagged value meta."""
    def __init__(self, name, bases, dict):
        r"""Init tagged value meta.

        Args:
            name (str): Name.
            bases (Any): Base.
            dict (dict): Dictionary.
        """
        for fn_name in self._proxies:
            try: getattr(self, fn_name)
            except AttributeError: setattr(self, fn_name, ProxyDelegate(fn_name, self._proxies[fn_name]))


class PassThroughProxy:
    r"""Pass through proxy."""
    def __init__(self, fn_name, obj):
        r"""Init pass through proxy.

        Args:
            fn_name (str): Name.
            obj (Any): Object.
        """
        self.fn_name = fn_name
        self.target = obj.proxy_target

    def __call__(self, *args):
        r"""Call operator.

        Args:
            args (list): Arguments.

        Returns:
            Callable[[list], Any]: Function.
        """
        fn = getattr(self.target, self.fn_name)
        ret = fn(*args)
        return ret


class ConvertArgsProxy(with_metaclass(PassThroughProxy, object)):
    r"""Convert arguments proxy."""
    def __init__(self, fn_name, obj):
        r"""Init convert arguments proxy.

        Args:
            fn_name (str): Name.
            obj (Any): Object.
        """
        PassThroughProxy.__init__(self, fn_name, obj)
        self.unit = obj.unit

    def __call__(self, *args):
        r"""Call operator.

        Args:
            args (list): Arguments.

        Returns:
            Callable[[list], Any]: Function.
        """
        converted_args = []
        for a in args:
            try: converted_args.append(a.convert_to(self.unit))
            except AttributeError: converted_args.append(TaggedValue(a, self.unit))
        converted_args = tuple([c.get_value() for c in converted_args])
        return PassThroughProxy.__call__(self, *converted_args)


class ConvertReturnProxy(PassThroughProxy):
    r"""Convert return proxy."""
    def __init__(self, fn_name, obj):
        r"""Init convert return proxy.

        Args:
            fn_name (str): Name.
            obj (Any): Object.
        """
        PassThroughProxy.__init__(self, fn_name, obj)
        self.unit = obj.unit

    def __call__(self, *args):
        r"""Call operator.

        Args:
            args (list): Arguments.

        Returns:
            Union[NotImplemented, TaggedValue]: Tagged value or error.
        """
        ret = PassThroughProxy.__call__(self, *args)
        return (NotImplemented if ret is NotImplemented else TaggedValue(ret, self.unit))


class ConvertAllProxy(PassThroughProxy):
    r"""Convert all proxy."""
    def __init__(self, fn_name, obj):
        r"""Init convert all proxy.

        Args:
            fn_name (str): Name.
            obj (Any): Object.
        """
        PassThroughProxy.__init__(self, fn_name, obj)
        self.unit = obj.unit

    def __call__(self, *args):
        r"""Call operator.

        Args:
            args (list): Arguments.

        Raises:
            TypeError: Bad unit.

        Returns:
            Union[NotImplemented, TaggedValue]: Tagged value.
        """
        converted_args = []
        arg_units = [self.unit]
        for a in args:
            if hasattr(a, 'get_unit') and not hasattr(a, 'convert_to'): return NotImplemented
            if hasattr(a, 'convert_to'):
                try: a = a.convert_to(self.unit)
                except Exception: raise TypeError("Error")
                arg_units.append(a.get_unit())
                converted_args.append(a.get_value())
            else:
                converted_args.append(a)
                if hasattr(a, 'get_unit'): arg_units.append(a.get_unit())
                else: arg_units.append(None)
        converted_args = tuple(converted_args)
        ret = PassThroughProxy.__call__(self, *converted_args)
        if ret is NotImplemented: return NotImplemented
        ret_unit = unit_resolver(self.fn_name, arg_units)
        if ret_unit is NotImplemented: return NotImplemented
        return TaggedValue(ret, ret_unit)


class TaggedValue(TaggedValueMeta):
    r"""Tagged value.

    Attributes:
        _proxies (Dict[str, Union[ConvertAllProxy, PassThroughProxy]): Operations.
    """

    _proxies = {'__add__': ConvertAllProxy,
                '__sub__': ConvertAllProxy,
                '__mul__': ConvertAllProxy,
                '__rmul__': ConvertAllProxy,
                '__cmp__': ConvertAllProxy,
                '__lt__': ConvertAllProxy,
                '__gt__': ConvertAllProxy,
                '__len__': PassThroughProxy}

    def __new__(cls, value, unit):
        r"""Generate new subclass.

        Args:
            value (TaggedValue): Value.
            unit (BasicUnit): Unit.

        Returns:
            TaggedValue: Tagged value.
        """
        value_class = type(value)
        try:
            subcls = type(f'TaggedValue_of_{value_class.__name__}', (cls, value_class,), {})
            if subcls not in units.registry: units.registry[subcls] = basicConverter
            return object.__new__(subcls)
        except TypeError:
            if cls not in units.registry: units.registry[cls] = basicConverter
            return object.__new__(cls)

    def __init__(self, value, unit):
        r"""Init tagged value.

        Args:
            value (Union[TaggedValue, Any]): Value
            unit (BasicUnit): Unit.
        """
        self.value = value
        self.unit = unit
        self.proxy_target = self.value

    def __getattribute__(self, name):
        r"""Get attribute.

        Args:
            name (str): Name.

        Returns:
            Any: Attribute for given name.
        """
        if name.startswith('__'): return object.__getattribute__(self, name)
        variable = object.__getattribute__(self, 'value')
        if hasattr(variable, name) and name not in self.__class__.__dict__: return getattr(variable, name)
        return object.__getattribute__(self, name)

    def __array__(self, dtype=object):
        r"""Get array.

        Args:
            dtype (type): Type to use.

        Returns:
            numpy.ndarray: Values.
        """
        return np.asarray(self.value).astype(dtype)

    def __array_wrap__(self, array, context):
        r"""Get wraped array.

        Args:
            array (Iterable[Any]): Array.
            context (Any): Context.

        Returns:
            TaggedValue: Tagged values in array.
        """
        return TaggedValue(array, self.unit)

    def __repr__(self):
        r"""Format status of instance.

        Returns:
            str: Formatted status of this instance.
        """
        return 'TaggedValue({!r}, {!r})'.format(self.value, self.unit)

    def __str__(self):
        r"""Get status of instance.

        Returns:
            str: Status of instance.
        """
        return str(self.value) + ' in ' + str(self.unit)

    def __len__(self):
        r"""Get length.

        Returns:
            int: Length.
        """
        return len(self.value)

    def __iter__(self):
        r"""Return a generator expression rather than use `yield`, so that.

        Raises:
            TypeError: is raised by iter(self) if appropriate when checking for iterability.

        Returns:
            Iterable[TaggedValue]: Iterator.
        """
        return (TaggedValue(inner, self.unit) for inner in self.value)

    def get_compressed_copy(self, mask):
        r"""Get compressed copy.

        Args:
            mask (numpy.ndarray): Mask array.

        Returns:
            TaggedValue: Value.
        """
        new_value = np.ma.masked_array(self.value, mask=mask).compressed()
        return TaggedValue(new_value, self.unit)

    def convert_to(self, unit):
        r"""Convert to units.

        Args:
            unit (BasicUnit): Unit.

        Returns:
            TaggedValue: Converted value.
        """
        if unit == self.unit or not unit: return self
        try: new_value = self.unit.convert_value_to(self.value, unit)
        except AttributeError: new_value = self
        return TaggedValue(new_value, unit)

    def get_value(self):
        r"""Get value.

        Returns:
            Any: Value.
        """
        return self.value

    def get_unit(self):
        r"""Get unit.

        Returns:
            BasicUnit: Unit
        """
        return self.unit


class BasicUnit:
    r"""Basic unit.

    Attributes:
        name (str): Name.
        fullname (str): Full name.
        conversions (dict): Conversions.
    """
    def __init__(self, name, fullname=None):
        r"""Init basic unit.

        Args:
            name (str): Name.
            fullname (Optional[str]): Full name.
        """
        self.name = name
        if fullname is None:
            fullname = name
        self.fullname = fullname
        self.conversions = dict()

    def __repr__(self):
        r"""Get formatted state of instance.

        Returns:
            str: Formatted state of instance.
        """
        return f'BasicUnit({self.name})'

    def __str__(self):
        r"""Get name.

        Returns:
            str: Name.
        """
        return self.fullname

    def __call__(self, value):
        r"""Call operator.

        Args:
            value (Any): Value.

        Returns:
            TaggedValue: Tagged value.
        """
        return TaggedValue(value, self)

    def __mul__(self, rhs):
        r"""Multiply operator.

        Args:
            rhs (Any): right side.

        Returns:
            Union[NotImplemented, TaggedValue]: Value.
        """
        value = rhs
        unit = self
        if hasattr(rhs, 'get_unit'):
            value = rhs.get_value()
            unit = rhs.get_unit()
            unit = unit_resolver('__mul__', (self, unit))
        if unit is NotImplemented:
            return NotImplemented
        return TaggedValue(value, unit)

    def __rmul__(self, lhs):
        r"""Multiply operator.

        Args:
            lhs (Any): Left side.

        Returns:
            TaggedValue: Multiplyed value.
        """
        return self * lhs

    def __array_wrap__(self, array, context):
        r"""Get wrapped array.

        Args:
            array (Iterable[Any]): Array of values.
            context (Any): Context.

        Returns:
            TaggedValue: Value.
        """
        return TaggedValue(array, self)

    def __array__(self, t=None, context=None):
        r"""Get array.

        Args:
            t (type): Type.
            context (Any): Context.

        Returns:
            numpy.ndarray: Array.
        """
        ret = np.array([1])
        if t is not None: return ret.astype(t)
        else: return ret

    def add_conversion_factor(self, unit, factor):
        r"""Add conversion factor.

        Args:
            unit (BasicUnit): Unit.
            factor (float): Factor.
        """
        def convert(x): return x * factor
        self.conversions[unit] = convert

    def add_conversion_fn(self, unit, fn):
        r"""Add conversion function.

        Args:
            unit (BasicUnit): Unit
            fn (Callable[[list], Any]): Function.
        """
        self.conversions[unit] = fn

    def get_conversion_fn(self, unit):
        r"""Get conversion function.

        Args:
            unit (BasicUnit): Unit.

        Returns:
            Callable[[list], Any]]: Function.
        """
        return self.conversions[unit]

    def convert_value_to(self, value, unit):
        r"""Convert values.

        Args:
            value (TaggedValue): Value.
            unit (BasicUnit): Unit.

        Returns:
            Any: Converted value.
        """
        conversion_fn = self.conversions[unit]
        ret = conversion_fn(value)
        return ret

    def get_unit(self):
        r"""Get unit.

        Returns:
            BasicUnit: Unit.
        """
        return self


class UnitResolver:
    r"""Unit resolver.

    Attributes:
        op_dict (Dict[str, Callable[[BasicUnit], Any]): Operations on unit.
    """
    def addition_rule(self, units):
        r"""Add rule for resolving.

        Args:
            units (BasicUnit): Unit.

        Returns:
            Union[NotImplemented, Any]: TODO.
        """
        for unit_1, unit_2 in zip(units[:-1], units[1:]):
            if unit_1 != unit_2:
                return NotImplemented
        return units[0]

    def multiplication_rule(self, units):
        r"""Multiplication rule.

        Args:
            units (BasicUnit): Unit.

        Returns:
            Union[NotImplemented, Any]: TODO.
        """
        non_null = [u for u in units if u]
        if len(non_null) > 1:
            return NotImplemented
        return non_null[0]

    op_dict = {
        '__mul__': multiplication_rule,
        '__rmul__': multiplication_rule,
        '__add__': addition_rule,
        '__radd__': addition_rule,
        '__sub__': addition_rule,
        '__rsub__': addition_rule}

    def __call__(self, operation, units):
        r"""Call operator.

        Args:
            operation (Callable[[BasicUnit], Any]): Operation.
            units (BasicUnit): Unit.

        Returns:
            Union[NotImplemented, Any]: TODO.
        """
        if operation not in self.op_dict:
            return NotImplemented

        return self.op_dict[operation](self, units)


unit_resolver = UnitResolver()

cm = BasicUnit('cm', 'centimeters')
inch = BasicUnit('inch', 'inches')
inch.add_conversion_factor(cm, 2.54)
cm.add_conversion_factor(inch, 1 / 2.54)

radians = BasicUnit('rad', 'radians')
degrees = BasicUnit('deg', 'degrees')
radians.add_conversion_factor(degrees, 180.0 / np.pi)
degrees.add_conversion_factor(radians, np.pi / 180.0)

secs = BasicUnit('s', 'seconds')
hertz = BasicUnit('Hz', 'Hertz')
minutes = BasicUnit('min', 'minutes')

secs.add_conversion_fn(hertz, lambda x: 1. / x)
secs.add_conversion_factor(minutes, 1 / 60.0)

# radians formatting
def rad_fn(x, pos=None):
    r"""Format for radians.

    Args:
        x (Any): Value.
        pos (Any): Position.

    Returns:
        str: Formatted string.
    """
    if x >= 0:
        n = int((x / np.pi) * 2.0 + 0.25)
    else:
        n = int((x / np.pi) * 2.0 - 0.25)

    if n == 0:
        return '0'
    elif n == 1:
        return r'$pi/2$'
    elif n == 2:
        return r'$pi$'
    elif n == -1:
        return r'$-pi/2$'
    elif n == -2:
        return r'$-pi$'
    elif n % 2 == 0:
        return fr'${n//2}pi$'
    else:
        return fr'${n}pi/2$'


class BasicUnitConverter(units.ConversionInterface):
    r"""Basic unit converter."""
    @staticmethod
    def axisinfo(unit, axis):
        r"""Axis info instance for x and unit.

        Args:
            unit (BasicUnit): Unit.
            axis (Any): Axis.

        Returns:
            Union[None, units.AxisInfo]: Axis info.
        """
        if unit == radians:
            return units.AxisInfo(
                majloc=ticker.MultipleLocator(base=np.pi / 2),
                majfmt=ticker.FuncFormatter(rad_fn),
                label=unit.fullname,
            )
        elif unit == degrees:
            return units.AxisInfo(
                majloc=ticker.AutoLocator(),
                majfmt=ticker.FormatStrFormatter(r'$%i^\circ$'),
                label=unit.fullname,
            )
        elif unit is not None:
            if hasattr(unit, 'fullname'):
                return units.AxisInfo(label=unit.fullname)
            elif hasattr(unit, 'unit'):
                return units.AxisInfo(label=unit.unit.fullname)
        return None

    @staticmethod
    def convert(val, unit, axis):
        r"""Convert value to unit.

        Args:
            val (Any): Value.
            unit (BasicUnit): Unit.
            axis (Any): Axis.

        Returns:
            Union[numpy.nan, Any]: Converted value.
        """
        if units.ConversionInterface.is_numlike(val):
            return val
        if np.iterable(val):
            if isinstance(val, np.ma.MaskedArray): val = val.astype(float).filled(np.nan)
            out = np.empty(len(val))
            for i, thisval in enumerate(val):
                if np.ma.is_masked(thisval): out[i] = np.nan
                else:
                    try: out[i] = thisval.convert_to(unit).get_value()
                    except AttributeError: out[i] = thisval
            return out
        if np.ma.is_masked(val): return np.nan
        else: return val.convert_to(unit).get_value()

    @staticmethod
    def default_units(x, axis):
        r"""Value of default unit for x or None.

        Args:
            x (Any): Value.
            axis (Any): Axis.

        Returns:
            BasicUnit: Default unit.
        """
        if np.iterable(x):
            for thisx in x: return thisx.unit
        return x.unit

def cos(x):
    r"""Cosinus.

    Args:
        x (Union[Iterable[float], float]): Value

    Returns:
        float: Cosinus value.
    """
    if np.iterable(x): return [np.cos(val.convert_to(radians).get_value()) for val in x]
    else: return np.cos(x.convert_to(radians).get_value())

basicConverter = BasicUnitConverter()
units.registry[BasicUnit] = basicConverter
units.registry[TaggedValue] = basicConverter
