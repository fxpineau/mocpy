# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from .py23_compat import range, int

import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import ICRS
from astropy import wcs
from astropy.table import Table

from astropy_healpix import HEALPix
from astropy_healpix.healpy import nside2npix

from .interval_set import IntervalSet
from .utils import uniq2orderipix, trailing_zeros

__author__ = "Thomas Boch, Matthieu Baumann"
__copyright__ = "CDS, Centre de Données astronomiques de Strasbourg"

__license__ = "BSD 3-Clause License"
__email__ = "thomas.boch@astro.unistra.fr, matthieu.baumann@astro.unistra.fr"


class MOC:
    """Multi-order Spatial Coverage.

    MOC stands for Mutli-Order Coverage. It is a spatial and hierarchical description of a region on a sphere.
    MOC is an IVOA standard and is the subject of a publication that you can find
    <here `http://www.ivoa.net/documents/MOC/`>__.

    MOCs are based on the HEALPix sky tessellation using the NESTED numbering scheme. A MOC is a set of
    HEALPix pixels at different orders with a maximum HEALPix resolution corresponding to the order 29 i.e. a cell
    resolution of ~393.2μas.

    * MOCs are usually stored as FITS file containing a list of UNIQ numbers describing HEALPix cells at different orders.
      This class aims at creating MOCs from FITS/json format, FITS image with a mask array,
      `astropy.coordinates.SkyCoord` and `astropy.units.Quantity`.

    * Basic operations on MOCs are available such as the intersection, union, difference, complement.

    * A :func:`~mocpy.moc.MOC.contains` method aims at filtering (ra, dec) positions expressed as
    `astropy.units.Quantity` through the MOC.

    * You can do MOC serialization to FITS (i.e. list of UNIQ numbers stored in a binary HDU table) and JSON.
      An optional parameter allows you to write it to a file.
    """
    HPY_MAX_NORDER = 29

    def __init__(self, interval_set=None):
        interval = IntervalSet() if interval_set is None else interval_set
        self._interval_set = interval

    def __eq__(self, another_moc):
        """
        Test equality between self and ``another_moc``

        Parameters
        ----------
        another_moc : `~mocpy.abstract_moc.AbstractMoc`
            the moc object to test the equality with

        Returns
        -------
        result : bool
            True if the interval sets of self and ``another_moc`` are equal (the interval sets are checked
            for consistency before comparing them).
        """
        if not isinstance(another_moc, MOC):
            raise TypeError('Cannot compare an AbstractMOC with a {0}'.format(type(another_moc)))

        return self._interval_set == another_moc._interval_set

    """
    MOC properties
    """
    @property
    def max_order(self):
        """
        This returns the deepest order needed to describe the current _interval_set
        """
        # TODO: cache value
        combo = int(0)
        for iv in self._interval_set._intervals:
            combo |= iv[0] | iv[1]

        ret = MOC.HPY_MAX_NORDER - (trailing_zeros(combo) // 2)
        if ret < 0:
            ret = 0

        return ret

    @property
    def sky_fraction(self):
        """
        return the sky fraction (between 0 and 1) covered by the MOC
        """
        pix_id_arr = self._best_res_pixels()
        nb_pix_filled = pix_id_arr.size
        return nb_pix_filled / float(3 << (2*(self.max_order + 1)))

    """
    Basic functions for manipulating MOCs.
    """
    def intersection(self, another_moc, *args):
        """
        Intersection between self and other mocs.

        Parameters
        ----------
        another_moc : `~mocpy.abstract_moc.AbstractMOC`
            the MOC/TimeMoc used for performing the intersection with self
        args : `~mocpy.abstract_moc.AbstractMOC`
            other MOCs

        Returns
        -------
        result : `~mocpy.moc.MOC` or `~mocpy.tmoc.TimeMoc`
            MOC object whose interval set corresponds to : self & ``moc``
        """
        interval_set = self._interval_set.intersection(another_moc._interval_set)
        for moc in args:
            interval_set = interval_set.intersection(moc._interval_set)

        return self.__class__(interval_set)

    def union(self, another_moc, *args):
        """
        Union between self and other mocs.

        Parameters
        ----------
        another_moc : `mocpy.abstract_moc.AbstractMOC`
            the MOC/TimeMoc to bind to self
        args : `~mocpy.abstract_moc.AbstractMOC`
            other MOCs

        Returns
        -------
        result : `~mocpy.moc.MOC` or `~mocpy.tmoc.TimeMoc`
            MOC object whose interval set corresponds to : self | ``moc``
        """
        interval_set = self._interval_set.union(another_moc._interval_set)
        for moc in args:
            interval_set = interval_set.union(moc._interval_set)

        return self.__class__(interval_set)

    def difference(self, another_moc, *args):
        """
        Difference between self and other mocs.

        Parameters
        ----------
        moc : `mocpy.abstract_moc.AbstractMOC`
            the MOC/TimeMoc to substract from self
        args : `~mocpy.abstract_moc.AbstractMOC`
            other MOCs

        Returns
        -------
        result : `~mocpy.moc.MOC` or `~mocpy.tmoc.TimeMoc`
            MOC object whose interval set corresponds to : self - ``moc``
        """
        interval_set = self._interval_set.difference(another_moc._interval_set)
        for moc in args:
            interval_set = interval_set.difference(moc._interval_set)

        return self.__class__(interval_set)

    def complement(self):
        """
        Create a mocpy object being the complemented of self

        Returns
        -------
        complement : `~mocpy.AbstractMoc`
            the complemented moc
        """
        res = []
        intervals_l = sorted(self._interval_set._intervals.tolist())

        if intervals_l[0][0] > 0:
            res.append((0, intervals_l[0][0]))

        last = intervals_l[0][1]

        for itv in intervals_l[1:]:
            res.append((last, itv[0]))
            last = itv[1]

        max_pix_order = 3 << 60

        if last < max_pix_order:
            res.append((last, max_pix_order))

        return self.__class__(IntervalSet(np.asarray(res)))

    def add_neighbours(self):
        """
        Add all the pixels at max order in the neighbourhood of the MOC:

        1. Get the HEALPix array of the MOC at the its max order.
        2. Get the HEALPix array containing the neighbors of the first array (it consists of an ``extended`` HEALPix
           array containing the first one).
        3. Compute the difference between the second and the first HEALPix array to get only the neighboring pixels
           located at the border of the MOC.
        4. This array of HEALPix neighbors are added to the MOC to get an ``extended`` MOC at its max order.

        Returns
        -------
        moc : `~mocpy.moc.MOC`
            self
        """
        pix_id_arr = self._best_res_pixels()

        hp = HEALPix(nside=(1 << self.max_order), order='nested')
        neighbour_pix_arr = MOC._get_neighbour_pix(hp, pix_id_arr)

        augmented_pix_arr = np.setdiff1d(neighbour_pix_arr, pix_id_arr)

        shift = 2 * (MOC.HPY_MAX_NORDER - self.max_order)
        intervals_arr = np.vstack((augmented_pix_arr << shift, (augmented_pix_arr + 1) << shift)).T

        self._interval_set = self._interval_set.union(IntervalSet.from_numpy_array(intervals_arr))
        return self

    def remove_neighbours(self):
        """
        Remove all the pixels at max order located at the bound of the moc:

        1. Get the HEALPix array of the MOC at the its max order.
        2. Get the HEALPix array containing the neighbors of the first array (it consists of an ``extended`` HEALPix
           array containing the first one).
        3. Compute the difference between the second and the first HEALPix array to get only the neighboring pixels
           located at the border of the MOC.
        4. Same as step 2 to get the HEALPix neighbors of the last computed array.
        5. The difference between the original MOC HEALPix array and this one gives a new MOC whose borders are removed.

        Returns
        -------
        moc : `~mocpy.moc.MOC`
            self
        """
        pix_id_arr = self._best_res_pixels()

        hp = HEALPix(nside=(1 << self.max_order), order='nested')
        neighbour_pix_arr = MOC._get_neighbour_pix(hp, pix_id_arr)

        only_neighbour_arr = np.setxor1d(neighbour_pix_arr, pix_id_arr)

        bound_pix_arr = MOC._get_neighbour_pix(hp, only_neighbour_arr)

        diminished_pix_arr = np.setdiff1d(pix_id_arr, bound_pix_arr)

        shift = 2 * (MOC.HPY_MAX_NORDER - self.max_order)
        intervals_arr = np.vstack((diminished_pix_arr << shift, (diminished_pix_arr + 1) << shift)).T
        self._interval_set = IntervalSet.from_numpy_array(intervals_arr)
        return self

    def degrade_to_order(self, new_order):
        """
        Degrade self to a mocpy object at max_order being equal to ``new_order``

        Parameters
        ----------
        new_order : int

        Returns
        -------
        moc : `mocpy.moc.MOC` or `mocpy.tmoc.TimeMoc`
            the res decreased mocpy object
        """
        shift = 2 * (MOC.HPY_MAX_NORDER - new_order)
        ofs = (int(1) << shift) - 1
        mask = ~ofs
        adda = int(0)
        addb = ofs
        iv_set = []

        for iv in self._interval_set._intervals:
            a = (iv[0] + adda) & mask
            b = (iv[1] + addb) & mask
            if b > a:
                iv_set.append((a, b))

        return self.__class__(IntervalSet.from_numpy_array(np.asarray(iv_set)))

    """
    Classmethods for instanciating a `~mocpy.moc.MOC` object
    """
    @classmethod
    def from_json(cls, json_moc):
        """
        Create a MOC from a dictionary of HEALPix arrays indexed by their order (in json format)

        Parameters
        ----------
        json_moc : {str : [int]}
            A dictionary of HEALPix arrays indexed by their order

        Returns
        -------
        moc : `~mocpy.moc.MOC` or `~mocpy.tmoc.TimeMoc`
            the MOC/TimeMoc object reflecting ``json_moc``.

        """
        intervals_arr = np.array([])
        for order, pix_l in json_moc.items():
            pix_arr = np.array(pix_l)
            p1 = pix_arr
            p2 = pix_arr + 1
            shift = 2 * (MOC.HPY_MAX_NORDER - int(order))

            itv_arr = np.vstack((p1 << shift, p2 << shift)).T
            if intervals_arr.size == 0:
                intervals_arr = itv_arr
            else:
                intervals_arr = np.vstack((intervals_arr, itv_arr))

        return cls(IntervalSet.from_numpy_array(intervals_arr))

    @classmethod
    def from_fits(cls, filename):
        """
        Load a MOC from a MOC fits file (i.e. a fits file in which pix are stored as a list of NUNIQ HEALPix numbers
        in a binary HDU table).

        Parameters
        ----------
        filename : str
            the path to the moc fits file

        Returns
        -------
        result : `~mocpy.moc.MOC` or `~mocpy.tmoc.TimeMoc`
            the mocpy object having as interval set the one stored in the fits file located at ``path``
        """
        table = Table.read(filename)

        intervals = np.vstack((table['UNIQ'], table['UNIQ']+1)).T

        nuniq_interval_set = IntervalSet.from_numpy_array(intervals)
        interval_set = IntervalSet.from_nuniq_interval_set(nuniq_interval_set)
        return cls(interval_set)

    @classmethod
    def from_image(cls, header, max_norder, mask_arr=None):
        """
        Create a `~mocpy.moc.MOC` from an image stored as a fits file

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            fits header containing all the info of where the image is located (position, size, etc...)
        max_norder : int
            the moc resolution
        mask_arr : `~numpy.ndarray`, optional
            a 2D boolean array of the same size of the image where pixels having the value 1 are part of
            the final MOC and pixels having the value 0 are not.

        Returns
        -------
        moc : `mocpy.MOC`
            the MOC object loaded from the ``mask_arr`` and ``header`` extracted from the image
        """
        # load the image data
        height = header['NAXIS2']
        width = header['NAXIS1']

        # use wcs from astropy to locate the image in the world coordinates
        w = wcs.WCS(header)

        if mask_arr is not None:
            # We have an array of pixels that are part of of survey
            y, x = np.where(mask_arr)
            pix_crd = np.dstack((x, y))[0]
        else:
            # If we do not have a mask array we create the moc of all the image
            #
            step_pix = 1
            """
            Coords returned by wcs_pix2world method correspond to pixel centers. We want to retrieve the moc pix
            crossing the borders of the image so we have to add 1/2 to the pixels coords before computing the lonlat.
            
            The step between two pix_crd is set to `step_pix` but can be diminished to have a better precision at the 
            borders so that all the image is covered (a too big step does not retrieve all
            the moc pix crossing the borders of the image).
            """
            x, y = np.mgrid[0.5:(width + 0.5 + step_pix):step_pix, 0.5:(height + 0.5 + step_pix):step_pix]
            pix_crd = np.dstack((x.ravel(), y.ravel()))[0]

        world_pix_crd = w.wcs_pix2world(pix_crd, 1)

        hp = HEALPix(nside=(1 << max_norder), order='nested', frame=ICRS())
        ipix = hp.lonlat_to_healpix(lon=world_pix_crd[:, 0] * u.deg,
                                    lat=world_pix_crd[:, 1] * u.deg)
        # remove doubles
        ipix = np.unique(ipix)

        shift = 2 * (MOC.HPY_MAX_NORDER - max_norder)
        intervals_arr = np.vstack((ipix << shift, (ipix + 1) << shift)).T

        # This MOC will be consistent when one will do operations on the moc (union, inter, ...) or
        # simply write it to a fits or json file
        interval_set = IntervalSet.from_numpy_array(intervals_arr)
        return cls(interval_set)

    @classmethod
    def from_fits_images(cls, path_l, max_norder):
        """
        Load a moc from a set of fits images

        Parameters
        ----------
        path_l : [str]
            the path list where the fits image are located
        max_norder : int
            moc resolution

        Returns
        -------
        moc : `~mocpy.MOC`
            the union of all the moc from path_l
        """
        moc = MOC()
        for path in path_l:
            header = fits.getheader(path)
            current_moc = MOC.from_image(header=header, max_norder=max_norder)
            moc = moc.union(current_moc)

        return moc

    @classmethod
    def from_skycoords(cls, skycoords, max_norder):
        """
        Create a MOC from a `astropy.coordinates.SkyCoord`.

        Parameters
        ----------
        skycoords : `astropy.coordinates.SkyCoord`
            a set of astropy skycoords
        max_norder : int
            the maximum order of the MOC

        Returns
        -------
        moc : `mocpy.MOC`
            a MOC of maximum order ``max_norder`` containing the ``skycoords``
        """
        hp = HEALPix(nside=(1 << max_norder), order='nested')
        ipix = hp.lonlat_to_healpix(skycoords.icrs.ra, skycoords.icrs.dec)

        shift = 2 * (MOC.HPY_MAX_NORDER - max_norder)
        intervals_arr = np.vstack((ipix << shift, (ipix + 1) << shift)).T

        interval_set = IntervalSet.from_numpy_array(intervals_arr)
        return cls(interval_set)

    @classmethod
    def from_lonlat(cls, lon, lat, max_norder):
        """
        Create a MOC from lon, lat `astropy.units.Quantity`.

        Parameters
        ----------
        lon : `astropy.units.Quantity`
            a set of ra quantities
        lat : `astropy.units.Quantity`
            a set of dec quantities
        max_norder : int
            the maximum order of the MOC

        Returns
        -------
        moc : `mocpy.MOC`
            a MOC of maximum order ``max_norder`` containing the positions defined by ``lon``, ``lat``.
        """
        hp = HEALPix(nside=(1 << max_norder), order='nested')
        ipix = hp.lonlat_to_healpix(lon, lat)

        shift = 2 * (MOC.HPY_MAX_NORDER - max_norder)
        intervals_arr = np.vstack((ipix << shift, (ipix + 1) << shift)).T

        interval_set = IntervalSet.from_numpy_array(intervals_arr)
        return cls(interval_set)

    """
    Filtering method
    """
    def contains(self, ra, dec, keep_inside=True):
        """
        Get a mask array (e.g. a numpy boolean array) of positions being inside (or outside) the
        mocpy object instance.

        Parameters
        ----------
        ra : `astropy.units.Quantity`
            right ascension array
        dec: `astropy.units.Quantity`
            declination array
        keep_inside : bool, optional
            True by default. If so the filtered table contains only observations that are located into
            the mocpy object. If ``keep_inside`` is False, the filtered table contains all observations lying outside
            the mocpy object.

        Returns
        -------
        array : `~numpy.darray`
            A mask boolean array
        """
        max_order = self.max_order
        m = np.zeros(nside2npix(1 << max_order), dtype=bool)

        pix_id_arr = self._best_res_pixels()
        m[pix_id_arr] = True

        if not keep_inside:
            m = np.logical_not(m)

        hp = HEALPix(nside=(1 << self.max_order), order='nested')
        pix_arr = hp.lonlat_to_healpix(ra, dec)

        return m[pix_arr]

    """
    MOC Plotting
    """
    def plot(self, title='MOC', coord='C'):
        """
        Plot the MOC object in a mollweide view

        This method uses matplotlib.

        Parameters
        ----------
        title : str
            the title of the plot
        coord : str
            type of coord (ICRS, Galactic, ...) in which the moc pix will be plotted.
            only ICRS coordinates are supported for the moment.
            TODO handle Galactic coordinates
        """
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.pyplot as plt

        plot_order = 8
        if self.max_order > plot_order:
            plotted_moc = self.degrade_to_order(plot_order)
        else:
            plotted_moc = self

        num_pixels_map = 768
        delta = 2 * np.pi / num_pixels_map

        x = np.arange(-np.pi, np.pi, delta)
        y = np.arange(-np.pi/2, np.pi/2, delta)
        lon_rad, lat_rad = np.meshgrid(x, y)

        pix_id_arr = plotted_moc._best_res_pixels()
        m = np.zeros(nside2npix(1 << plotted_moc.max_order))
        m[pix_id_arr] = 1

        hp = HEALPix(nside=(1 << plotted_moc.max_order), order='nested', frame=ICRS())

        pix_map = hp.lonlat_to_healpix(lon_rad * u.rad, lat_rad * u.rad)
        z = np.flip(m[pix_map], axis=1)

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection="mollweide")
        ax.set_xticklabels(['150°', '120°', '90°', '60°', '30°', '0°', '330°', '300°', '270°', '240°', '210°', '180°'])

        color_map = LinearSegmentedColormap.from_list('w2r', ['#eeeeee', '#aa0000'])
        color_map.set_under('w')
        color_map.set_bad('gray')

        ax.pcolormesh(x, y, z, cmap=color_map, vmin=0, vmax=1)
        ax.tick_params(labelsize=14, labelcolor='#000000')
        plt.title(title)
        plt.grid(True, linestyle='--', linewidth=1, color='#555555')

        plt.show()

    """
    Serialization methods
    """
    @staticmethod
    def _to_json(uniq_arr):
        """
        Serialize a mocpy object (array of uniq) to json

        Parameters
        ----------
        uniq_arr : `~numpy.ndarray`
            the array of uniq reprensenting the mocpy object to serialize

        Returns
        -------
        result_json : {str : [int]}
            a dictionary of pixel list each indexed by their order
        """
        result_json = {}

        order_arr, ipix_arr = uniq2orderipix(uniq_arr)
        min_order = order_arr[0]
        max_order = order_arr[-1]

        for order in range(min_order, max_order+1):
            pix_index = np.where(order_arr == order)[0]
            if pix_index.size:
                # there are pixels belonging to the current order
                ipix_order_arr = ipix_arr[pix_index]
                result_json[str(order)] = ipix_order_arr.tolist()

        return result_json

    @staticmethod
    def _to_fits(uniq_arr, moc_order, optional_kw_dict=None):
        """
        Serialize a mocpy object (array of uniq) to a fits format

        Parameters
        ----------
        uniq_arr : `numpy.ndarray`
            the array of uniq representing the mocpy object to serialize
        moc_order : int
            the order of the MOC described by ``uniq_arr``
        optional_kw_dict : dict
            optional keywords arguments added to the fits header

        Returns
        -------
        thdulist : `astropy.io.fits.HDUList`
            the fits serialization of the MOC/TimeMoc object
        """
        if moc_order <= 13:
            fits_format = '1J'
        else:
            fits_format = '1K'

        tbhdu = fits.BinTableHDU.from_columns(
            fits.ColDefs([fits.Column(name='UNIQ', format=fits_format, array=uniq_arr)]))
        tbhdu.header['PIXTYPE'] = 'HEALPIX'
        tbhdu.header['ORDERING'] = 'NUNIQ'
        tbhdu.header['COORDSYS'] = ('C', 'reference frame (C=ICRS)')
        tbhdu.header['MOCORDER'] = moc_order
        tbhdu.header['MOCTOOL'] = 'MOCPy'
        if optional_kw_dict:
            for key in optional_kw_dict:
                tbhdu.header[key] = optional_kw_dict[key]

        thdulist = fits.HDUList([fits.PrimaryHDU(), tbhdu])
        return thdulist

    def write(self, path=None, format='fits', optional_kw_dict=None, write_to_file=False):
        """
        Serialize a MOC/TimeMoc object.

        Possibility to write it to a file at ``path``. Format can be 'fits' or 'json',
        though only the fits format is officially supported by the IVOA.

        Parameters
        ----------
        path : str, optional
            path to save the MOC object. The mocpy is written to path only if ``serialize`` is False. None by default
        format : str, optional
            format in which the mocpy object will be serialized. Constraint to takes its value
            among "fits" or "json". By default, ``format`` is set to "fits".
        optional_kw_dict : {str, _}, optional
            optional dictionary keywords for the header of the fits file. Only used if ``format`` is "fits"
        write_to_file : bool, optional
            Set to False by default. In this case, this method does not write to a file but returns the serialized form
            of the MOC/TimeMoc object to the user. If you want to write to a file

        Returns
        -------
        result : a `astropy.io.fits.HDUList` if ``format`` is set to "fits" or {str, [int]} otherwise
            The serialization of the MOC/TimeMoc object
        """
        formats = ('fits', 'json')
        if format not in formats:
            raise ValueError('format should be one of %s' % (str(formats)))

        # Get all the uniq number from the nuniq intervals
        intervals_uniq_l = IntervalSet.to_nuniq_interval_set(self._interval_set)._intervals
        uniq_l = []
        for uniq_iv in intervals_uniq_l:
            for uniq in range(uniq_iv[0], uniq_iv[1]):
                uniq_l.append(uniq)

        uniq_arr = np.asarray(uniq_l)

        if format == 'fits':
            result = self.__class__._to_fits(uniq_arr=uniq_arr,
                                             moc_order=self.max_order,
                                             optional_kw_dict=optional_kw_dict)
            if write_to_file:
                result.writeto(path, overwrite=True)
        else:
            # json format serialization
            result = self.__class__._to_json(uniq_arr=uniq_arr)
            if write_to_file:
                import json
                with open(path, 'w') as h:
                    h.write(json.dumps(result, sort_keys=True, indent=2))

        return result

    """
    Utility methods
    """
    def _best_res_pixels(self):
        """
        Get a numpy array of all the HEALPix indexes contained in the MOC at its max order.

        Returns
        -------
        array : `numpy.ndarray`
            the array of HEALPix at ``max_order``
        """
        factor = 2 * (MOC.HPY_MAX_NORDER - self.max_order)
        pix_l = []
        for iv in self._interval_set._intervals:
            for val in range(iv[0] >> factor, iv[1] >> factor):
                pix_l.append(val)

        return np.asarray(pix_l)

    @staticmethod
    def _get_neighbour_pix(hp, pix_arr):
        """
        Get all the pixels neighbours of ``pix_arr``

        Parameters
        ----------
        hp : `~astropy_healpix.HEALPix`
            the HEALPix context
        pix_arr : `~numpy.ndarray`
            the input array of pixels
        Returns
        -------
        neighbour_pix_arr : `~numpy.ndarray`
            an array of pixels containing the neighbours of the pixels in ``pix_arr``
        """
        neighbour_pix_arr = np.unique(hp.neighbours(pix_arr).ravel())
        # Remove negative pixel values returned by `~astropy_healpix.HEALPix.neighbours`
        return neighbour_pix_arr[np.where(neighbour_pix_arr >= 0)]