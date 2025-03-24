"""Reduced COO class

Taken from :mod:`sparse.COO` and modified.
BSD 3-Clause License: https://github.com/pydata/sparse/blob/main/LICENSE

Our version strips the numba dependency and is compatible with Python 3.13.
"""

from _operator import mul
from functools import reduce
from math import isnan as math_isnan
from numbers import Integral, Number
from typing import Iterable

from numpy import (
    asarray,
    stack,
    zeros,
    uint64,
    where,
    ones,
    bool as np_bool,
    min_scalar_type,
    empty,
    sum as np_sum,
    uint8,
    diff,
    append,
    nonzero,
    add,
    dtype,
    prod,
    argsort,
    multiply,
    array,
    any as np_any,
    unique,
    arange,
    concatenate,
    flatnonzero,
    isnan,
    ndarray,
    asanyarray,
    issubdtype,
    integer,
    nonzero as np_nonzero,
    intp,
    allclose,
    isclose,
)
from numpy.lib.mixins import NDArrayOperatorsMixin


class COOreduced(NDArrayOperatorsMixin):
    """
    A sparse multidimensional array.

    Reduced Functionality, only supporting indices with possible duplicates,
    and no data (all data is assumed to be 1).

    Code adapted from the :cls:`COO <sparse.COO>` class in sparse.

    BSD 3-Clause License: https://github.com/pydata/sparse/blob/main/LICENSE
    """

    def __init__(
        self, coords, data=None, shape=None, has_duplicates=True, sorted=False
    ):
        if data is None:
            if not coords:
                data = []
                coords = []

            # [((i, j, k), value), (i, j, k), value), ...]
            elif isinstance(coords[0][0], Iterable):
                if coords:
                    assert len(coords[0]) == 2
                data = [x[1] for x in coords]
                coords = [x[0] for x in coords]
                coords = asarray(coords).T

            # (data, (row, col, slab, ...))
            else:
                data = coords[0]
                coords = stack(coords[1], axis=0)

        self.data = asarray(data)
        self.coords = asarray(coords)

        if self.coords.ndim == 1:
            self.coords = self.coords[None, :]

        if shape and not self.coords.size:
            self.coords = zeros((len(shape), 0), dtype=uint64)

        if shape is None:
            if self.coords.nbytes:
                shape = tuple((self.coords.max(axis=1) + 1).tolist())
            else:
                shape = ()

        if not isinstance(shape, Iterable):
            shape = (int(shape),)

        if not all(isinstance(l, Integral) and int(l) >= 0 for l in shape):
            raise ValueError(
                "shape must be an non-negative integer or a tuple "
                "of non-negative integers."
            )

        self.shape = tuple(int(l) for l in shape)

        # if self.shape:
        # dtype = min_scalar_type(max(max(self.shape) - 1, 0))
        dtype = uint64
        # else:
        #     dtype = uint8

        self.coords = self.coords.astype(dtype)
        assert not self.shape or len(self.data) == self.coords.shape[1]
        self.has_duplicates = has_duplicates
        self.sorted = sorted
        self.sum_duplicates()

    def nonzero(self):
        """
        Get the indices where this array is nonzero.

        Returns
        -------
        idx : tuple[numpy.ndarray]
            The indices where this array is nonzero.
        """
        return tuple(self.coords)

    @property
    def ndim(self):
        """
        The number of dimensions of this array.

        Returns
        -------
        int
            The number of dimensions of this array.
        """
        return len(self.shape)

    @property
    def dtype(self):
        """
        The datatype of this array.

        Returns
        -------
        numpy.dtype
            The datatype of this array.

        See Also
        --------
        numpy.ndarray.dtype : Numpy equivalent property.
        scipy.sparse.coo_matrix.dtype : Scipy equivalent property.
        """
        return self.data.dtype

    @property
    def nnz(self):
        """
        The number of nonzero elements in this array. Note that any duplicates in
        :code:`coords` are counted multiple times. To avoid this, call :obj:`COO.sum_duplicates`.

        Returns
        -------
        int
            The number of nonzero elements in this array.

        See Also
        --------
        DOK.nnz : Equivalent :obj:`DOK` array property.
        numpy.count_nonzero : A similar Numpy function.
        scipy.sparse.coo_matrix.nnz : The Scipy equivalent property.
        """
        return self.coords.shape[1]

    @property
    def size(self):
        """
        The number of all elements (including zeros) in this array.

        Returns
        -------
        int
            The number of elements.

        See Also
        --------
        numpy.ndarray.size : Numpy equivalent property.
        """
        return reduce(mul, self.shape, 1)

    def __getitem__(self, index):  # making object subscriptable
        if not isinstance(index, tuple):
            if isinstance(index, str):
                data = self.data[index]
                idx = where(data)
                coords = list(self.coords[:, idx[0]])
                coords.extend(idx[1:])

                return COOreduced(
                    coords,
                    data[idx].flatten(),
                    shape=self.shape + self.data.dtype[index].shape,
                    has_duplicates=self.has_duplicates,
                    sorted=self.sorted,
                )
            else:
                index = (index,)

        last_ellipsis = len(index) > 0 and index[-1] is Ellipsis
        index = normalize_index(index, self.shape)
        if len(index) != 0 and all(
            not isinstance(ind, Iterable) and ind == slice(None) for ind in index
        ):
            return self
        mask = ones(self.nnz, dtype=np_bool)
        for i, ind in enumerate([i for i in index if i is not None]):
            if not isinstance(ind, Iterable) and ind == slice(None):
                continue
            mask &= _mask(self.coords[i], ind, self.shape[i])

        n = mask.sum()
        coords = []
        shape = []
        i = 0
        for ind in index:
            if isinstance(ind, Integral):
                i += 1
                continue
            elif isinstance(ind, slice):
                step = ind.step if ind.step is not None else 1
                if step > 0:
                    start = ind.start if ind.start is not None else 0
                    start = max(start, 0)
                    stop = ind.stop if ind.stop is not None else self.shape[i]
                    stop = min(stop, self.shape[i])
                    if start > stop:
                        start = stop
                    shape.append((stop - start + step - 1) // step)
                else:
                    start = ind.start or self.shape[i] - 1
                    stop = ind.stop if ind.stop is not None else -1
                    start = min(start, self.shape[i] - 1)
                    stop = max(stop, -1)
                    if start < stop:
                        start = stop
                    shape.append((start - stop - step - 1) // (-step))

                dt = min_scalar_type(
                    min(-(dim - 1) if dim != 0 else -1 for dim in shape)
                )
                coords.append((self.coords[i, mask].astype(dt) - start) // step)
                i += 1
            elif isinstance(ind, Iterable):
                old = self.coords[i][mask]
                new = empty(shape=old.shape, dtype=old.dtype)
                for j, item in enumerate(ind):
                    new[old == item] = j
                coords.append(new)
                shape.append(len(ind))
                i += 1
            elif ind is None:
                coords.append(zeros(n))
                shape.append(1)

        for j in range(i, self.ndim):
            coords.append(self.coords[j][mask])
            shape.append(self.shape[j])

        if coords:
            coords = stack(coords, axis=0)
        else:
            if last_ellipsis:
                coords = empty((0, np_sum(mask)), dtype=uint8)
            else:
                if np_sum(mask) != 0:
                    return self.data[mask][0]
                else:
                    return _zero_of_dtype(self.dtype)[()]
        shape = tuple(shape)
        data = self.data[mask]

        return COOreduced(
            coords,
            data,
            shape=shape,
            has_duplicates=self.has_duplicates,
            sorted=self.sorted,
        )

    def todense(self):
        """
        Convert this :obj:`COO` array to a dense :obj:`numpy.ndarray`. Note that
        this may take a large amount of memory if the :obj:`COO` object's :code:`shape`
        is large.

        Returns
        -------
        numpy.ndarray
            The converted dense array.

        See Also
        --------
        DOK.todense : Equivalent :obj:`DOK` array method.
        scipy.sparse.coo_matrix.todense : Equivalent Scipy method.

        Examples
        --------
        >>> x = np.random.randint(100, size=(7, 3))
        >>> s = COO.from_numpy(x)
        >>> x2 = s.todense()
        >>> np.array_equal(x, x2)
        True
        """
        self.sum_duplicates()
        x = zeros(shape=self.shape, dtype=self.dtype)

        coords = tuple([self.coords[i, :] for i in range(self.ndim)])
        data = self.data

        if coords != ():
            x[coords] = data
        else:
            if len(data) != 0:
                x[coords] = data

        return x

    def sum_duplicates(self):
        """
        Sums data corresponding to duplicates in :obj:`COO.coords`.

        See Also
        --------
        scipy.sparse.coo_matrix.sum_duplicates : Equivalent Scipy function.
        """
        # Inspired by scipy/sparse/coo.py::sum_duplicates
        # See https://github.com/scipy/scipy/blob/master/LICENSE.txt
        if not self.has_duplicates and self.sorted:
            return
        if not self.coords.size:
            return

        self.sort_indices()

        linear = self.linear_loc()
        unique_mask = diff(linear) != 0

        if unique_mask.sum() == len(unique_mask):  # already unique
            self.has_duplicates = False
            return

        unique_mask = append(True, unique_mask)

        coords = self.coords[:, unique_mask]
        (unique_inds,) = nonzero(unique_mask)
        data = add.reduceat(self.data, unique_inds, dtype=self.data.dtype)

        self.data = data
        self.coords = coords
        self.has_duplicates = False

    def reduce(self, method, axis=None, keepdims=False, **kwargs):
        """
        Performs a reduction operation on this array.

        Parameters
        ----------
        method : numpy.ufunc
            The method to use for performing the reduction.
        axis : Union[int, Iterable[int]], optional
            The axes along which to perform the reduction. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        kwargs : dict
            Any extra arguments to pass to the reduction operation.

        Returns
        -------
        COO
            The result of the reduction operation.

        Raises
        ------
        ValueError
            If reducing an all-zero axis would produce a nonzero result.

        Notes
        -----
        This function internally calls :obj:`COO.sum_duplicates` to bring the array into
        canonical form.

        See Also
        --------
        numpy.ufunc.reduce : A similar Numpy method.
        COO.nanreduce : Similar method with ``NaN`` skipping functionality.

        Examples
        --------
        You can use the :obj:`COO.reduce` method to apply a reduction operation to
        any Numpy :code:`ufunc`.

        >>> x = np.ones((5, 5), dtype=np.int)
        >>> s = COO.from_numpy(x)
        >>> s2 = s.reduce(np.add, axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([5, 5, 5, 5, 5])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        reduction.

        >>> s3 = s.reduce(np.add, axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        You can also pass in any keyword argument that :obj:`numpy.ufunc.reduce` supports.
        For example, :code:`dtype`. Note that :code:`out` isn't supported.

        >>> s4 = s.reduce(np.add, axis=1, dtype=np.float16)
        >>> s4.dtype
        dtype('float16')

        By default, this reduces the array down to one number, reducing along all axes.

        >>> s.reduce(np.add)
        25
        """
        zero_reduce_result = method.reduce([_zero_of_dtype(self.dtype)], **kwargs)

        if zero_reduce_result != _zero_of_dtype(dtype(zero_reduce_result)):
            raise ValueError(
                "Performing this reduction operation would produce "
                "a dense result: %s" % str(method)
            )

        # Needed for more esoteric reductions like product.
        self.sum_duplicates()

        if axis is None:
            axis = tuple(range(self.ndim))

        if not isinstance(axis, tuple):
            axis = (axis,)

        if set(axis) == set(range(self.ndim)):
            result = method.reduce(self.data, **kwargs)
            if self.nnz != self.size:
                result = method(result, _zero_of_dtype(self.dtype)[()], **kwargs)
        else:
            axis = tuple(axis)
            neg_axis = tuple(ax for ax in range(self.ndim) if ax not in axis)

            a = self.transpose(neg_axis + axis)
            a = a.reshape(
                (
                    prod([self.shape[d] for d in neg_axis]),
                    prod([self.shape[d] for d in axis]),
                )
            )
            a.sort_indices()

            result, inv_idx, counts = _grouped_reduce(
                a.data, a.coords[0], method, **kwargs
            )
            missing_counts = counts != a.shape[1]
            result[missing_counts] = method(
                result[missing_counts], _zero_of_dtype(self.dtype), **kwargs
            )
            coords = a.coords[0:1, inv_idx]
            a = COOreduced(
                coords, result, shape=(a.shape[0],), has_duplicates=False, sorted=True
            )

            a = a.reshape([self.shape[d] for d in neg_axis])
            result = a

        if keepdims:
            result = _keepdims(self, result, axis)
        return result

    def sum(self, axis=None, keepdims=False, dtype=None, out=None):
        """
        Performs a sum operation along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to sum. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        COO
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.sum` : Equivalent numpy function.
        scipy.sparse.coo_matrix.sum : Equivalent Scipy function.
        :obj:`nansum` : Function with ``NaN`` skipping.

        Notes
        -----
        * This function internally calls :obj:`COO.sum_duplicates` to bring the array into
          canonical form.
        * The :code:`out` parameter is provided just for compatibility with Numpy and
          isn't actually supported.

        Examples
        --------
        You can use :obj:`COO.sum` to sum an array across any dimension.

        >>> x = np.ones((5, 5), dtype=np.int)
        >>> s = COO.from_numpy(x)
        >>> s2 = s.sum(axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([5, 5, 5, 5, 5])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        sum.

        >>> s3 = s.sum(axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        You can pass in an output datatype, if needed.

        >>> s4 = s.sum(axis=1, dtype=np.float16)
        >>> s4.dtype
        dtype('float16')

        By default, this reduces the array down to one number, summing along all axes.

        >>> s.sum()
        25
        """
        assert out is None
        return self.reduce(add, axis=axis, keepdims=keepdims, dtype=dtype)

    def sort_indices(self):
        """
        Sorts the :obj:`COO.coords` attribute. Also sorts the data in
        :obj:`COO.data` to match.
        """
        if self.sorted:
            return

        linear = self.linear_loc(signed=True)

        if (diff(linear) > 0).all():  # already sorted
            self.sorted = True
            return

        order = argsort(linear)
        self.coords = self.coords[:, order]
        self.data = self.data[order]
        self.sorted = True

    def linear_loc(self, signed=False):
        """
        The nonzero coordinates of a flattened version of this array. Note that
        the coordinates may be out of order.

        Parameters
        ----------
        signed : bool, optional
            Whether to use a signed datatype for the output array. :code:`False`
            by default.

        Returns
        -------
        numpy.ndarray
            The flattened coordinates.

        See Also
        --------
        :obj:`numpy.flatnonzero` : Equivalent Numpy function.

        Examples
        --------
        >>> x = np.eye(5)
        >>> s = COO.from_numpy(x)
        >>> s.linear_loc()  # doctest: +NORMALIZE_WHITESPACE
        array([ 0,  6, 12, 18, 24], dtype=uint8)
        >>> np.array_equal(np.flatnonzero(x), s.linear_loc())
        True
        """
        n = reduce(mul, self.shape, 1)
        if signed:
            n = -n
        dtype = min_scalar_type(n)
        out = zeros(self.coords.shape[1], dtype=dtype)
        tmp = zeros(self.coords.shape[1], dtype=dtype)
        strides = 1
        for i, d in enumerate(self.shape[::-1]):
            multiply(self.coords[-(i + 1), :], strides, out=tmp, dtype=dtype)
            add(tmp, out, out=out)
            strides *= d
        return out

    def reshape(self, shape):
        """
        Returns a new :obj:`COO` array that is a reshaped version of this array.

        Parameters
        ----------
        shape : tuple[int]
            The desired shape of the output array.

        Returns
        -------
        COO
            The reshaped output array.
        """
        if self.shape == shape:
            return self
        if any(d == -1 for d in shape):
            extra = int(self.size / prod([d for d in shape if d != -1]))
            shape = tuple([d if d != -1 else extra for d in shape])

        if self.shape == shape:
            return self

        # TODO: this self.size enforces a 2**64 limit to array size
        linear_loc = self.linear_loc()

        max_shape = max(shape) if len(shape) != 0 else 1
        coords = empty((len(shape), self.nnz), dtype=min_scalar_type(max_shape - 1))
        strides = 1
        for i, d in enumerate(shape[::-1]):
            coords[-(i + 1), :] = (linear_loc // strides) % d
            strides *= d
        result = COOreduced(
            coords,
            data=self.data,
            shape=shape,
            has_duplicates=self.has_duplicates,
            sorted=self.sorted,
        )

        return result

    def transpose(self, axes=None):
        """
        Returns a new array which has the order of the axes switched.

        Parameters
        ----------
        axes : Iterable[int], optional
            The new order of the axes compared to the previous one. Reverses the axes
            by default.

        Returns
        -------
        COO
            The new array with the axes in the desired order.

        See Also
        --------
        :obj:`COO.T` : A quick property to reverse the order of the axes.
        numpy.ndarray.transpose : Numpy equivalent function.

        """
        if axes is None:
            axes = list(reversed(range(self.ndim)))

        # Normalize all axe indices to posivite values
        axes = array(axes)
        axes[axes < 0] += self.ndim

        if np_any(axes >= self.ndim) or np_any(axes < 0):
            raise ValueError("invalid axis for this array")

        if len(unique(axes)) < len(axes):
            raise ValueError("repeated axis in transpose")

        if not len(axes) == self.ndim:
            raise ValueError("axes don't match array")

        # Normalize all axe indices to posivite values
        try:
            axes = arange(self.ndim)[list(axes)]
        except IndexError:
            raise ValueError("invalid axis for this array")

        if len(unique(axes)) < len(axes):
            raise ValueError("repeated axis in transpose")

        if not len(axes) == self.ndim:
            raise ValueError("axes don't match array")

        axes = tuple(axes)

        if axes == tuple(range(self.ndim)):
            return self

        shape = tuple(self.shape[ax] for ax in axes)
        result = COOreduced(
            self.coords[axes, :],
            self.data,
            shape,
            has_duplicates=self.has_duplicates,
        )

        return result

    @property
    def T(self):
        """
        Returns a new array which has the order of the axes reversed.

        Returns
        -------
        COO
            The new array with the axes in the desired order.

        See Also
        --------
        :obj:`COO.transpose` : A method where you can specify the order of the axes.
        numpy.ndarray.T : Numpy equivalent property.
        """
        return self.transpose(tuple(range(self.ndim))[::-1])


def _zero_of_dtype(dtype):
    return zeros((), dtype=dtype)


def _grouped_reduce(x, groups, method, **kwargs):
    """
    Performs a :code:`ufunc` grouped reduce.

    Parameters
    ----------
    x : np.ndarray
        The data to reduce.
    groups : np.ndarray
        The groups the data belongs to. The groups must be
        contiguous.
    method : np.ufunc
        The :code:`ufunc` to use to perform the reduction.
    kwargs : dict
        The kwargs to pass to the :code:`ufunc`'s :code:`reduceat`
        function.

    Returns
    -------
    result : np.ndarray
        The result of the grouped reduce operation.
    inv_idx : np.ndarray
        The index of the first element where each group is found.
    counts : np.ndarray
        The number of elements in each group.
    """
    # Partial credit to @shoyer
    # Ref: https://gist.github.com/shoyer/f538ac78ae904c936844
    flag = concatenate(([True] if len(x) != 0 else [], groups[1:] != groups[:-1]))
    inv_idx = flatnonzero(flag)
    result = method.reduceat(x, inv_idx, **kwargs)
    counts = diff(concatenate((inv_idx, [len(x)])))
    return result, inv_idx, counts


def _keepdims(original, new, axis):
    shape = list(original.shape)
    for ax in axis:
        shape[ax] = 1
    return new.reshape(shape)


def normalize_index(idx, shape):
    """Normalize slicing indexes
    1.  Replaces ellipses with many full slices
    2.  Adds full slices to end of index
    3.  Checks bounding conditions
    4.  Replaces numpy arrays with lists
    5.  Posify's integers and lists
    6.  Normalizes slices to canonical form
    Examples
    --------
    >>> normalize_index(1, (10,))
    (1,)
    >>> normalize_index(-1, (10,))
    (9,)
    >>> normalize_index([-1], (10,))
    (array([9]),)
    >>> normalize_index(slice(-3, 10, 1), (10,))
    (slice(7, None, None),)
    >>> normalize_index((Ellipsis, None), (10,))
    (slice(None, None, None), None)
    """
    if not isinstance(idx, tuple):
        idx = (idx,)
    idx = replace_ellipsis(len(shape), idx)
    n_sliced_dims = 0
    for i in idx:
        if hasattr(i, "ndim") and i.ndim >= 1:
            n_sliced_dims += i.ndim
        elif i is None:
            continue
        else:
            n_sliced_dims += 1
    idx = idx + (slice(None),) * (len(shape) - n_sliced_dims)
    if len([i for i in idx if i is not None]) > len(shape):
        raise IndexError("Too many indices for array")

    none_shape = []
    i = 0
    for ind in idx:
        if ind is not None:
            none_shape.append(shape[i])
            i += 1
        else:
            none_shape.append(None)

    for i, d in zip(idx, none_shape):
        if d is not None:
            check_index(i, d)
    idx = tuple(map(sanitize_index, idx))
    idx = tuple(map(normalize_slice, idx, none_shape))
    idx = posify_index(none_shape, idx)
    return idx


def replace_ellipsis(n, index):
    """Replace ... with slices, :, : ,:
    >>> replace_ellipsis(4, (3, Ellipsis, 2))
    (3, slice(None, None, None), slice(None, None, None), 2)
    >>> replace_ellipsis(2, (Ellipsis, None))
    (slice(None, None, None), slice(None, None, None), None)
    """
    # Careful about using in or index because index may contain arrays
    isellipsis = [i for i, ind in enumerate(index) if ind is Ellipsis]
    if not isellipsis:
        return index
    elif len(isellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    else:
        loc = isellipsis[0]
    extra_dimensions = n - (len(index) - sum(i is None for i in index) - 1)
    return (
        index[:loc] + (slice(None, None, None),) * extra_dimensions + index[loc + 1 :]
    )


def _mask(coords, idx, shape):
    if isinstance(idx, Integral):
        return coords == idx
    elif isinstance(idx, slice):
        step = idx.step if idx.step is not None else 1
        if step > 0:
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else shape
            return (coords >= start) & (coords < stop) & (coords % step == start % step)
        else:
            start = idx.start if idx.start is not None else (shape - 1)
            stop = idx.stop if idx.stop is not None else -1
            return (coords <= start) & (coords > stop) & (coords % step == start % step)
    elif isinstance(idx, Iterable):
        mask = zeros(len(coords), dtype=bool)
        for item in idx:
            mask |= _mask(coords, item, shape)
        return mask


def check_index(ind, dimension):
    """Check validity of index for a given dimension
    Examples
    --------
    >>> check_index(3, 5)
    >>> check_index(5, 5)
    Traceback (most recent call last):
    ...
    IndexError: Index is not smaller than dimension 5 >= 5
    >>> check_index(6, 5)
    Traceback (most recent call last):
    ...
    IndexError: Index is not smaller than dimension 6 >= 5
    >>> check_index(-1, 5)
    >>> check_index(-6, 5)
    Traceback (most recent call last):
    ...
    IndexError: Negative index is not greater than negative dimension -6 <= -5
    >>> check_index([1, 2], 5)
    >>> check_index([6, 3], 5)
    Traceback (most recent call last):
    ...
    IndexError: Index out of bounds 5
    >>> check_index(slice(0, 3), 5)
    """
    # unknown dimension, assumed to be in bounds
    if isnan(dimension):
        return
    elif isinstance(ind, (list, ndarray)):
        x = asanyarray(ind)
        if issubdtype(x.dtype, integer) and (
            (x >= dimension).any() or (x < -dimension).any()
        ):
            raise IndexError("Index out of bounds %s" % dimension)
        elif x.dtype == bool and len(x) != dimension:
            raise IndexError(
                "boolean index did not match indexed array; dimension is %s "
                "but corresponding boolean dimension is %s",
                (dimension, len(x)),
            )
    elif isinstance(ind, slice):
        return
    elif ind is None:
        return

    elif ind >= dimension:
        raise IndexError(
            "Index is not smaller than dimension %d >= %d" % (ind, dimension)
        )

    elif ind < -dimension:
        msg = "Negative index is not greater than negative dimension %d <= -%d"
        raise IndexError(msg % (ind, dimension))


def sanitize_index(ind):
    """Sanitize the elements for indexing along one axis
    >>> sanitize_index([2, 3, 5])
    array([2, 3, 5])
    >>> sanitize_index([True, False, True, False])
    array([0, 2])
    >>> sanitize_index(np.array([1, 2, 3]))
    array([1, 2, 3])
    >>> sanitize_index(np.array([False, True, True]))
    array([1, 2])
    >>> type(sanitize_index(np.int32(0))) # doctest: +SKIP
    <type 'int'>
    >>> sanitize_index(1.0)
    1
    >>> sanitize_index(0.5)
    Traceback (most recent call last):
    ...
    IndexError: Bad index.  Must be integer-like: 0.5
    """
    if ind is None:
        return None
    elif isinstance(ind, slice):
        return slice(
            _sanitize_index_element(ind.start),
            _sanitize_index_element(ind.stop),
            _sanitize_index_element(ind.step),
        )
    elif isinstance(ind, Number):
        return _sanitize_index_element(ind)
    index_array = asanyarray(ind)
    if index_array.dtype == bool:
        nonzero = np_nonzero(index_array)
        if len(nonzero) == 1:
            # If a 1-element tuple, unwrap the element
            nonzero = nonzero[0]
        return asanyarray(nonzero)
    elif issubdtype(index_array.dtype, integer):
        return index_array
    elif issubdtype(index_array.dtype, float):
        int_index = index_array.astype(intp)
        if allclose(index_array, int_index):
            return int_index
        else:
            check_int = isclose(index_array, int_index)
            first_err = index_array.ravel()[flatnonzero(~check_int)[0]]
            raise IndexError("Bad index.  Must be integer-like: %s" % first_err)
    else:
        raise TypeError("Invalid index type", type(ind), ind)


def _sanitize_index_element(ind):
    """Sanitize a one-element index."""
    if isinstance(ind, Number):
        ind2 = int(ind)
        if ind2 != ind:
            raise IndexError("Bad index.  Must be integer-like: %s" % ind)
        else:
            return ind2
    elif ind is None:
        return None
    else:
        raise TypeError("Invalid index type", type(ind), ind)


def normalize_slice(idx, dim):
    """Normalize slices to canonical form
    Parameters
    ----------
    idx: slice or other index
    dim: dimension length
    Examples
    --------
    >>> normalize_slice(slice(0, 10, 1), 10)
    slice(None, None, None)
    """

    if isinstance(idx, slice):
        start, stop, step = idx.start, idx.stop, idx.step
        if start is not None:
            if start < 0 and not math_isnan(dim):
                start = max(0, start + dim)
            elif start > dim:
                start = dim
        if stop is not None:
            if stop < 0 and not math_isnan(dim):
                stop = max(0, stop + dim)
            elif stop > dim:
                stop = dim

        step = 1 if step is None else step

        if step > 0:
            if start == 0:
                start = None
            if stop == dim:
                stop = None
        else:
            if start == dim - 1:
                start = None
            if stop == -1:
                stop = None

        if step == 1:
            step = None
        return slice(start, stop, step)
    return idx


def posify_index(shape, ind):
    """Flip negative indices around to positive ones
    >>> posify_index(10, 3)
    3
    >>> posify_index(10, -3)
    7
    >>> posify_index(10, [3, -3])
    array([3, 7])
    >>> posify_index((10, 20), (3, -3))
    (3, 17)
    >>> posify_index((10, 20), (3, [3, 4, -3]))  # doctest: +NORMALIZE_WHITESPACE
    (3, array([ 3,  4, 17]))
    """
    if isinstance(ind, tuple):
        return tuple(map(posify_index, shape, ind))
    if isinstance(ind, Integral):
        if ind < 0 and not math_isnan(shape):
            return ind + shape
        else:
            return ind
    if isinstance(ind, (ndarray, list)) and not math_isnan(shape):
        ind = asanyarray(ind)
        return where(ind < 0, ind + shape, ind)

    return ind


def asnumpy(a, dtype=None, order=None):
    """Returns a dense numpy array from an COOreduced array.

    Parameters
    ----------
    a : COOreduced
        Array in COOreduced format.
    order: ({'C', 'F', 'A'})
        The desired memory layout of the output
        array. When ``order`` is 'A', it uses 'F' if ``a`` is
        fortran-contiguous and 'C' otherwise.

    Returns
    -------
    numpy.ndarray: Converted array on the host memory.
    """
    return asarray(a.todense(), dtype=dtype, order=order)
