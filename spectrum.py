import warnings

import numpy as np
from scipy.interpolate import interp1d
# could implement fallback to np.interp is scipy is not available

from astropy import table
import astropy.units as u
import astropy.constants as const
from astropy.extern import six

# These are functions in order to emphasize that they do stuff to spectra,
# it's not a property of the spectrum.
# Or put everything for single spectrum un spectrum?
# bin_up is not for flux spectrum.
# should use np.sum() for count number spectrum -> Inheritance diagram
# instead of slice disp, should I just support that in __getitem__ (if inout is a 
# wavelength) or is that too magical?
# currently use doppler_optical - make different options (e.g. variable)


def coadd_simple(spectra, dispersion=None, **kwargs):
    '''A simple way to coadd several spectra

    All spectra are interpolated to `dispersion` and for each wavelengths bin the
    mean over all n spectra is calculated. The reported uncertainty is just the mean
    of all uncertainties scaled by 1/sqrt(n).

    Parameters
    ----------
    spectra : list of ~COSspectrum instances
        spectra to be averaged
    dispersion : ~astropy.quantity.Quantity
        dispersion axis of the new spectrum. If `None` the dispersion axis of the 
        first spectrum in `spectra` is used.
    
    All other parameters are passed to spectrum.interpol.

    Returns
    -------
    spec : COSspectrum

    See also
    --------
    coadd_errorweighted (not implemented yet)
    '''
    if dispersion is None:
        dispersion = spectra[0].disp
    fluxes = np.zeros((len(spectra), len(dispersion)))
    # since numpy operation will destroy the flux unit, need to convert here by hand 
    # until that is fixed (fluxes can be NDdata once that works with quantities)
    fluxunit = spectra[0].flux.unit
    errors = np.zeros_like(fluxes)
    for i, s in enumerate(spectra):
        s_new = s.interpol(dispersion, **kwargs)
        fluxes[i,:] = s_new.flux.to(fluxunit)
        if (errors is None) or s.uncertainty is None:
            errors = None
        else:
            errors[i,:] = s_new.error.to(fluxunit)

    fluxes = fluxes.mean(axis=0) * fluxunit
    if errors is None:
        return COSspectrum(data=[dispersion, fluxes],
                           names=[spectra[0].dispersion, 'FLUX'],
                           dispersion=spectra[0].dispersion,
                           )
    else:
        errors =  errors.mean(axis=0)/np.sqrt(len(spectra))*fluxunit
        return COSspectrum(data=[dispersion, fluxes, errors],
                           names=[spectra[0].dispersion, 'FLUX', spectra[0].uncertainty], 
                       dispersion=spectra[0].dispersion, uncertainty=spectra[0].uncertainty,
                           )


class COSspectrum(table.Table):
    '''
    To make more general:
    make "dispersion" or something similar and allow to set that as an alias for
    what I currently call wavelength
    '''
    fluxname = 'FLUX'  # default to None, here, make setter in init for general
    dispersion = None  # set to default 'WAVELENGTH' for specific COSspectrum
    uncertainty = None  # can be removed when uncertainties are part of NDData

    def __init__(self, *args, **kwargs):
        if 'dispersion' in kwargs:
            self.dispersion = kwargs.pop('dispersion')
        if 'uncertainty' in kwargs:
            self.uncertainty = kwargs.pop('uncertainty')

        super(COSspectrum, self).__init__(*args, **kwargs)

    def _new_from_slice(self, slice_):
        spec = super(COSspectrum, self)._new_from_slice(slice_)
        spec.fluxname = self.fluxname
        spec.dispersion = self.dispersion
        spec.uncertainty = self.uncertainty
        return spec

    def __getitem__(self, item):
        spec = super(COSspectrum, self).__getitem__(item)
        if isinstance(item, (tuple, list)) and all(isinstance(x, six.string_types)
                                                     for x in item):
            spec.fluxname = self.fluxname
            spec.dispersion = self.dispersion
            spec.uncertainty = self.uncertainty
        return spec


    @classmethod
    def read(cls, filename):
        '''specific to COS - can be made into a Reader'''
        data = []
        
        tab = Table.read(filename)
        for c in tab.columns:
            if len(tab[c].shape) == 2:
                data.append(table.Column(data=tab[c].data.flatten(),
                                         unit=tab[c].unit,
                                         name=c))
        flattab = cls(data, meta=tab.meta, dispersion='WAVELENGTH')
        # COS is not an echelle spectrograph, so there never is any overlap
        flattab.sort('WAVELENGTH')
        return flattab

    @property
    def flux(self):
        if self.fluxname is None:
            raise ValueError('Need to set spectrum.fluxname="NAME".')
        return self[self.fluxname].data * self[self.fluxname].unit


    @property
    def disp(self):
        # Currently table columns are not quantities.
        # see issue astropy/#2486
        if self.dispersion is None:
            raise ValueError('Need to set spectrum.dispersion="NAME".')
        return self[self.dispersion].data * self[self.dispersion].unit

    @property
    def error(self):
        # Can be removed when NDData really supports uncertainties
        if self.uncertainty is None:
            raise ValueError('Need to set spectrum.uncertainty="NAME".')
        return self[self.uncertainty].data * self[self.uncertainty].unit


    def _slice_disp(self, bounds, equivalencies = []):
        b0 = bounds[0].to(self.disp.unit, equivalencies=equivalencies)
        b1 = bounds[1].to(self.disp.unit, equivalencies=equivalencies)
        ind = (self.disp >= b0) & (self.disp <= b1)
        return self[ind]


    def slice_disp(self, bounds):
        '''Return a portion of the spectrum between given dispersion values

        Parameters
        ----------
        bounds : list of two quantities
            [lower bound, upper bound] in dispersion of spectrally equivalent unit
        
        Returns
        -------
        spec : COSSpectrum
            spectrum that is limited to the range from bound[0] to bound[1]

        See also
        --------
        slice_rv
        '''
        equil = u.spectral()
        return self._slice_disp(bounds, equivalencies=equil)


    def slice_rv(self, bounds, rest):
        '''Return a portion of the spectrum between given radial velocity values

        Parameters
        ----------
        bounds : list of two quantities
            [lower bound, upper bound] in radial velocity
        rest : astropy.quantity.Quantity
            Rest wavelength/frequency of spectral feature
        
        Returns
        -------
        spec : COSSpectrum
            spectrum that is limited to the range from bound[0] to bound[1]

        See also
        --------
        slice_disp
        '''
        equil = u.spectral()
        equil.extend(u.doppler_optical(rest))
        return self._slice_disp(bounds, equivalencies=equil)


    def normalize(self, model, **kwargs):
        #fit model and normalize flux and error
        # Makes sense here or have user interact with flux directly?
        raise NotImplementedError


    def shift_rv(self, rv):
        '''Shift spectrum by rv
        
        Parameters
        ----------
        rv : astropy.quantity.Quantity
            radial velocity
        '''
        self[self.dispersion] = (self.disp.to(u.m, equivalencies=u.spectral()) * (
                1.*u.dimensionless_unscaled-rv/const.c)).to(
                self.disp.unit, equivalenvies=u.spectral()).value


    # overload add, substract, devide to interpol automatically?
        
    def bin_up(self, factor, other_cols={}):
        '''bin up an array by factor factor
   
        If the number of elements in spectrum is not n * factor with n=1,2,3,...
        the remaining bins at the end are discarded.
        By itself, this function knows how to deal with dispersion, flux and uncertainty.
        The parameter `other_cols` can be used to specify how to bin up other columns in 
        the spectrum. Columns with no rule for binning up are discarded.
   
        Parameters
        ----------
        x : array
            data to be binned
        factor : integer
            binning factor
        other_cols : dictionary
            The keys in this dictionary are the names of columns in the spectrum, its values
            are functions that define how to bin up other columns (e.g. for a column that holds
            a mask or data quality flag it does not make sense to calculate the mean).
            The function is called on an array of shape [N, factor] and needs to return 
            a [N] array. For example `other_cols={'quality': lambda x: np.max(x, axis=1)}`
            would assign the new bin the maximum of all quality values in the original bins
            that contribute to the new bin.

        Returns
        -------
        spec : Spectrum
            A new spectrum object.
        '''
        n = len(self) // factor
        # makes a copy incl. all metadata and column formats
        # Which cols to keep?
        keepcols = set([x for x in [self.dispersion, self.fluxname, self.uncertainty] if x is not None])
        keepcols = keepcols.union(set(other_cols.keys()))
        spec = self[list(keepcols)][::factor]
        spec[self.dispersion] = (self.disp[:n*factor].reshape((n, factor))).mean(axis=1)
        if self.uncertainty is None:
            spec[self.fluxname] = (self.flux[:n*factor].reshape((n, factor))).mean(axis=1)
        else:
            f, e = np.ma.average(self.flux[:n*factor].reshape((n, factor)), weights=1./(self.error[:n*factor].reshape((n, factor)))**2., axis=1, returned=True)
            spec[self.fluxname] = f
            spec[self.uncertainty] = (1./e)**0.5
        for col in other_cols:
            spec[col] = other_cols[col](self[col][:n*factor].reshape((n, factor)))
        return spec


    def interpol(self, new_dispersion, **kwargs):
        new_disp = new_dispersion.to(self.disp.unit, equivalencies=u.spectral())

        f_flux = interp1d(self.disp, self.flux, **kwargs)
        newflux = f_flux(new_disp)

        names = [self.dispersion, self.fluxname]
        vals = [new_disp, newflux]

        if self.uncertainty is not None:
            warnings.warn('The uncertainty column is interpolated.' + 
                         'Bins are no longer independent and might require scaling.' +
                         'It is up to the user the decide if the uncertainties are still meaningful.')
            names.append(self.uncertainty)
            f_uncert = interp1d(self.disp, self.error, **kwargs)
            vals.append(f_uncert(new_disp))

        # To-Do: Add other columns that should be interpolated to names, vals here
        # Add switch or keyword to select them

        newcols = []
        for name, val in zip(names, vals):
            col = self[name]
            newcols.append(col.__class__(data=val, name=name, 
                                         description=col.description, unit=col.unit,
                                         format=col.format, meta=col.meta))
        return self.__class__(newcols, meta=self.meta, dispersion=self.dispersion, 
                              uncertainty=self.uncertainty)
        
