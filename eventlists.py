'''collect all functions that deal directly with event lists for e.g.::
    - Timing
'''
import numpy as np
import scipy
import scipy.stats
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from PyAstronomy.pyTiming import pyPeriod
from astropy.table import Table
import glob
import pickle
import os

from detrend_poly import detrend_poly


def tabmax(tab, key, bin=1):
    '''bin a table in `key` and return position of largest bin

    Use this to find e.g. the stongest line in an event list.
    '''
    minx = min(tab[key])
    maxx = max(tab[key])
    hist, bins = np.histogram(tab[key], bins=(maxx - minx) / bin)
    amax = np.argmax(hist)
    return np.mean(bins[amax:amax + 2])


'''To Do:
    - determin deadtime automatically from header values
'''


class CosCounts(object):
    '''COS eventlist data

    This oject holds a single COS eventlist.
    It performs deadtime corrections and provides methods to generate
    descriptors of the dataset, e.g. calculate a Lomb-Scargle periodogram.
    '''
    clock_cycle = 0.032

    def __init__(self, filename, nsum=1, deadtime=None, CIV=False):
        '''generate some descriptive statistic from event list

        Parameters
        ----------
        filename : string
        nsum : integer
            rebinning factor for primitive statistics
            When rebinned with `nsum`, then make bins so that they all have
            same number of time steps, because the binning
            introduces artificial frequencies / autocorrelation otherwise
            Use `nsum` to bin up, so that the count number in each bin
            is Gaussian distributed.
        '''
        tab = Table.read(filename, hdu=1)
        # make unfiltered lc. That's important for the dead-time correction
        hist, bins = np.histogram(tab['TIME'] / self.clock_cycle,
                                  bins=np.arange(-.5, max(tab['TIME'] / self.clock_cycle), nsum))
        self.full_lc = hist
        # FUV
        if 'rawtag_' in filename:
            ypos = tabmax(tab, 'RAWY', bin=10)
            tab = tab[(tab['RAWY'] > ypos - 40) & (tab['RAWY'] < ypos + 40)]
        # NUV
        else:
            tab = tab[tab['RAWY'] < 500]
        # C IV is brightest line
        if CIV:
            xpos = tabmax(tab, 'RAWX', bin=100)
            tab = tab[(tab['RAWX'] > xpos - 250) & (tab['RAWX'] < xpos + 500)]

        hist, bins = np.histogram(tab['TIME'] / self.clock_cycle,
                                  bins=np.arange(-.5, max(tab['TIME'] / self.clock_cycle), nsum))
        self.hist = hist
        self.bins = bins
        self.deadtime = deadtime
        self.clock_cycle *= nsum

    def LombScargle(self):
        '''find significance of highest peak in COS event list

        This routine bins an event list in 0.032s indervals (the clock time)
        and runs a Lomb-Scargle periodogram.
        It return the false alarm probability (FAP) of the highest peak.
        '''
        # LombScargle expects even number of entries
        start = self.hist.shape[0] % 2
        lc = pyPeriod.TimeSeries(self.bins[start + 1:],
                                 mlab.detrend_linear(self.hist[start:]))
        self.ls = pyPeriod.LombScargle(lc, ofac=1, hifac=1)
        # ignore the lowest freq (= the length of the dataset)
        return self.ls.FAP(max(self.ls.power[3:])), max(self.ls.power[3:])

    def stats(self):
        '''generate some descriptive statistice from event list

        Returns
        -------
        n : integer
            number of elements in lightcurve
        mean : float
        detrend_var : float
            variance of detrended lightcurve
        kurtosis : float
            should be 0 for a Gaussian distribution
        slope : float
            slope of a linear regression
        p : float
            two-sided p-value for a hypothesis test whose null hypothesis is
            that the slope is zero.
        '''

        slope, intercept, r, p, stderr = scipy.stats.linregress(self.bins[1:], self.hist)
        n, lim, mean, var, skew, kurt = scipy.stats.describe(self.hist)

        return len(self.hist), mean, var, kurt, slope, p

    def deadtime_corr(self, nsum=[1]):
        '''return mean and var for deadtime corrected lightcurves

        Caution: Any event on the detector will start the dead-time,
        even if it comes from the wavecal lamps!
        '''
        mean_dead_time = []
        var_dead_time = []
        for i, ns in enumerate(nsum):
            timebin = self.clock_cycle * ns
            # bin up as required
            # ignore end of array if len(hist)/nsum is not an integer
            end = len(self.hist) / int(ns) * int(ns)
            hist = np.reshape(self.hist[:end],(-1,ns)).sum(axis = 1)
            full_lc = np.reshape(self.full_lc[:end],(-1,ns)).sum(axis = 1)
            dead_time_corr_lc = hist / (1.-(full_lc/timebin)*self.deadtime)
            var_dead_time.append(scipy.var(detrend_poly(dead_time_corr_lc, deg = 3)))
            mean_dead_time.append(np.mean(dead_time_corr_lc))

        return np.array(mean_dead_time), np.array(var_dead_time)

    def acorr(self):
        '''generate some descriptive statistice from event list

        Returns
        -------
        slope : float
            slope of a linear regression to positive lags.
            I expect a significantly negative slope for interesting autocorrelations functions.
        '''
        lags, c, line, lines = plt.acorr(self.hist, detrend = mlab.detrend_mean, maxlags = 500)
        linregress = scipy.stats.linregress(lags[lags > 0], c[lags > 0])
        return linregress[0]


def analyse_COS_counts(COS, tab, i, do_FAP=True):
    '''calcualte some statistical description for a COS object

    i.e. calculate LS, var, mean, ... , and deadtime corrected quantities
    Place the results in line i of the tab table

    Parameters
    ---------
    COS : CosCounts object
    tab : tab_COS_counts_analysis table
    i : integer
        line in tab which will hold the results
    '''
    if do_FAP:
        tab.RAP[i], tab.LSPower[i] = COS.LombScargle()
    tab.n[i], tab.mean[i], tab.var[i], tab.kurt[i], tab.slope[i], tab.p[i] = COS.stats()
    tab.slopeacorr[i] = COS.acorr()
    tab.deadtimemean[i,:], tab.deadtimevar[i,:] = COS.deadtime_corr(nsum = tab.nsum)


def tab_COS_counts_analysis(globfile):
    '''make an atpy Table with empty columns for values
    '''
    tab = atpy.Table()
    tab.nsum = np.array(2.**np.arange(12)) # from small to large scales
    tab.add_column('file', glob.glob(globfile))
    tab.add_empty_column('FAP', dtype = np.float)
    tab.add_empty_column('LSPower', dtype = np.float)
    tab.add_empty_column('n', dtype = np.float)
    tab.add_empty_column('mean', dtype = np.float)
    tab.add_empty_column('var', dtype = np.float)
    tab.add_empty_column('kurt', dtype = np.float)
    tab.add_empty_column('slope', dtype = np.float)
    tab.add_empty_column('p', dtype = np.float)
    tab.add_empty_column('slopeacorr', dtype = np.float)
    tab.add_empty_column('deadtimemean', dtype = np.float, shape = (len(tab), len(tab.nsum)))
    tab.add_empty_column('deadtimevar', dtype = np.float, shape = (len(tab), len(tab.nsum)))
    return tab


def analyse_all_COS_evts(datapath, do_FAP = True):
    '''call analyse_COS_counts for all eventlists in datapath
    '''
    tuvb = tab_COS_counts_analysis(datapath+'*_rawtag_b.fits')
    tciv = tab_COS_counts_analysis(datapath+'*_rawtag_b.fits')
    tuva = tab_COS_counts_analysis(datapath+'*_rawtag_a.fits')
    tnuv = tab_COS_counts_analysis(datapath+'*_rawtag.fits')

    for i, spec in enumerate(tuvb['file']):
        analyse_COS_counts(CosCounts(str(spec), deadtime = 7.4e-6), tuvb, i, do_FAP = do_FAP)
        analyse_COS_counts(CosCounts(str(spec), deadtime = 7.4e-6, CIV = True), tciv, i, do_FAP = do_FAP)

    for i, spec in enumerate(tuva['file']):
        analyse_COS_counts(CosCounts(str(spec), deadtime = 7.4e-6), tuva, i, do_FAP = do_FAP)

    for i, spec in enumerate(tnuv['file']):
        analyse_COS_counts(CosCounts(str(spec), deadtime = 2.8e-7), tnuv, i, do_FAP = do_FAP)

    return tnuv, tuva, tuvb, tciv

### Plot scripts ###

def plot_FAPs(tnuv, tuva, tuvb, tciv):

    FAP = np.hstack([[0.],tnuv.FAP, tuva.FAP, tuvb.FAP, tciv.FAP]) # add 0 point for plotting purposes
    FAP.sort()
    y = np.arange(0,len(FAP), dtype= np.float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(FAP, y/max(y), lw=3.)
    ax.set_xlabel('FAP')
    ax.set_ylabel('cumulative distribution')
    pd.plotfile(fig, 'FAPs')
    return fig, ax

class extranoise(object):
    def __init__(self, sample1, sample2):
        self.sample1 = sample1
        self.sample2 = sample2
    def __call__(self, x, p):
        '''find the x where sample1 and x*sample2 could be drawn from
        the same parent distribution according to a KS test with exactly
        significance p

        x : ndarray of len 1

        p : float
            probability
        '''
        ksout = scipy.stats.ks_2samp(self.sample1 * x, self.sample2)
        #to find only x > 1
        penalty = 0 if x > 0. else 1. + (- x) *10.
        return np.abs(ksout[1] - p) + penalty

def plot_sigma(tnuv, tuva, tuvb, tciv, multsim = 1):
    #add a 0 so that the plot starts on the xaxis?
    var = np.vstack([tnuv.deadtimevar, tuva.deadtimevar, tuvb.deadtimevar, tciv.deadtimevar])
    mean = np.vstack([tnuv.deadtimemean, tuva.deadtimemean, tuvb.deadtimemean, tciv.deadtimemean])
    n = np.hstack([tnuv.n, tuva.n, tuvb.n, tciv.n])
    deadtime = np.hstack([np.tile(2.8e-7, len(tnuv)), np.tile(7.4e-6, len(tuva) + len(tuvb) + len(tciv))])

    '''Simulations take a long time. Open from file if these simlations
    exist, otherwise calculate autoematically and save result'''
    try:
        with open(os.path.join(pd.simpath,'simmean'+str(multsim)+'.dump')) as simfile:
            simmean = pickle.load(simfile)
        with open(os.path.join(pd.simpath,'simvar'+str(multsim)+'.dump')) as  simfile:
            simvar = pickle.load(simfile)

    except IOError:
        print 'No pre-computed lcs found.'
        print 'Simulating lcs. This make take a long time...'
        # just input the same values multiple times in the MC sims,
        # thus increasing the MC number and keeping the input close to obs
        meansiminput = np.tile(mean[:,0], multsim)
        nsiminput = np.tile(n, multsim)
        simdeadtime = np.tile(deadtime, multsim)
        simmean, simvar = sim_deadtime(nsiminput * 0.032, meansiminput/0.032, simdeadtime, nsum = tnuv.nsum)
        with open(os.path.join(pd.simpath,'simmean'+str(multsim)+'.dump'), 'w') as simfile:
            pickle.dump(simmean, simfile)
        with open(os.path.join(pd.simpath,'simvar'+str(multsim)+'.dump'), 'w') as  simfile:
            pickle.dump(simvar, simfile)

    #Plot 1 : chi^2 for nsum = 1
    rat = var[:,0]/mean[:,0]
    simrat = simvar[:,0]/simmean[:,0]
    rat.sort()
    simrat.sort()

    fig, ax = plot_variancechi2(rat, simrat, n)

    #plot2 : limit on extra variability
    extranoisefac = np.array(tnuv.nsum, dtype = np.float)
    ksobssim = np.array(tnuv.nsum, dtype = np.float)

    for i in range(len(extranoisefac)):
        rat = var[:,i]/mean[:,i]
        rat.sort()
        simrat = simvar[:,i] / simmean[:,i]
        simrat.sort()
        extranoisefac[i] = scipy.optimize.fmin(extranoise(rat, simrat), [1.005], args = [.01])
        temp, ksobssim[i] = scipy.stats.ks_2samp(rat, simrat)

    # calculate extranoisefac as ratio to the poissonnoise
    extranoisefac = np.sqrt(extranoisefac - 1.)
    # calc extranoisefac as ratio to flux
    extranoisefracofflux = extranoisefac / np.median(mean, axis = 0)
    fig2, ax2 = plot_sigmathreshold(tnuv.nsum, extranoisefracofflux)

    return fig, ax, fig2, ax2

def plot_sigmathreshold(nsum, extranoisefracofflux):
    ind = ((nsum * 0.032) < 150.)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogx(nsum * 0.032,extranoisefracofflux , lw = 3, label='data')
    ax.set_xlabel('time bin [s]')
    ax.set_ylabel('threshold')
    pd.plotfile(fig, 'sigmathreshold')
    return fig, ax

def plot_variancechi2(rat, ratsim, n):
    x = np.arange(1,max(n)*5.,5.)
    y = np.arange(0,len(rat), dtype= np.float)
    simy = np.arange(0,len(ratsim), dtype= np.float)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rat, y/max(y), 'o', mew = 2, label='data')
    ax.plot(ratsim, simy/max(simy), lw = 3, label='simulation')
    ax.plot(x/np.median(n), scipy.stats.chi2.cdf(x,np.median(n)-5), 'k:',lw = 3, label = r'$\chi^2$ distribution')
    ax.set_xlim(0.95, 1.1)
    ax.legend(loc = 'lower right')
    ax.set_xlabel(r'red. $\chi^2$')
    ax.set_ylabel('cumulative distribution')
    pd.plotfile(fig, 'sigmas')
    return fig, ax




class SimLc(object):
    timebin = .032

    def __init__(self, time, countrate, deadtime = None):
        self.deadtime = deadtime
        self.TIME_0 = np.random.random_sample(time*countrate) * time
        self.TIME_0.sort()
        self.TIME = self.apply_dead_time(self.TIME_0)
        self.TIME = self.apply_time_resolution(self.TIME)

    def apply_dead_time(self, TIME):
        diff = np.zeros_like(TIME)
        #alkways keep first photon
        diff[0] = np.inf
        diff[1:] = np.diff(TIME)
        return TIME[diff > self.deadtime]

    def apply_time_resolution(self, TIME):
        bins = np.arange(-self.timebin/2., max(TIME)+self.timebin*2., self.timebin)
        dig = np.digitize(TIME, bins)
        return dig * self.timebin

def sim_deadtime(time, countrate, deadtime, nsum = [1]):
    '''simulate a sample of lightcurves

    For each lightcurve the length and the true countrate are given.
    The photon arrival times are randomnly distributed, than events within
    the dead-time are discarded.
    (This uses an approximate algorithm, which assumes that at most one
    photon arrived during the dead time. This approximation should be valid
    for low count rates.)
    The lightcurve is than binned to the instrumental clock cycle.
    This data is treated with the same procedures as the observed data to
    obtain dead-time corrected mean and var values.

    Parameters
    ----------
    time : array_like
        time lenght (in s) for each lightcurve
    countrate : array_like
        countrate (in cts/s) for each lightcurve
    deadtime : array_like
        deadtime (in s) for each lightcurve
    nsum : array_like, optional
        list of nsum parameters

    Returns
    -------
    mean : ndarray
        dead-time corrected mean for each lc
    var : ndarray
        dead-time corrected variance for each lc
    '''
    mean = np.zeros([len(time), len(nsum)])
    var = np.zeros_like(mean)
    if len(time) != len(countrate):
        raise ValueError('time and countrate need same number of elements')
    if len(time) != len(deadtime):
        raise ValueError('time and countrate need same number of elements')
    for i in range(len(time)):
        lc = SimLc(time[i], countrate[i], deadtime[i])
        mean[i,:], var[i,:] = COS_counts_noise(lc, nsum = nsum)
    return mean, var

def plot_periodogram(hist, bins):
    def lspow(sig):
        return scipy.optimize.fmin(lambda x : np.abs(ls.FAP(x)-sig),[10.], disp = False)
    # LombScargle expects even number of entries
    start = hist.shape[0]%2
    lc = pyPeriod.TimeSeries(bins[start+1:], mlab.detrend_linear(hist[start:]))
    ls = pyPeriod.LombScargle(lc, ofac=1, hifac=1)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(ls.freq, ls.power, lw = 2.)
    #ax.plot(ax.get_xlim(), lspow(0.1) * np.array(1., 1., dtype = np.float), 'k:' )
    #ax.plot(ax.get_xlim(), lspow(0.01) * np.array(1., 1., dtype = np.float), 'k:')
    fig, ax = ls.plot(lw = 2., FAPlevels = [.1, 0.01])
    #ax.set_ylabel('Scargle Power')
    ax.set_xlabel('Frequency [Hz]')
    pd.plotfile(fig, 'LSperiodogram')
    ax.set_yscale('log')
    ax.set_xscale('log')
    pd.plotfile(fig, 'LSloglog')
    return fig, ax

def plot_lc(hist, bins):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(bins[:-1], hist, lw = 1.)
    ax.set_ylabel('Count rate [cts]')
    ax.set_xlabel('Time [s]')
    ax.set_xlim([0,max(bins)])
    pd.plotfile(fig, 'lcsingle')
    return fig, ax

def plot_acorr(hist, bins):
    lag, acorr, temp1, temp2 = plt.acorr(hist, normed = True, detrend = mlab.detrend_linear, maxlags = None)
    lag = np.array(lag, dtype = np.float)*0.032
    ind = (lag >=0)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(lag[ind], acorr[ind])
    ax.set_ylabel('Autocorrelation coefficient')
    ax.set_xlabel('Time scale [s]')
    ax.set_ylim(-0.03, +0.03)
    ax.set_xlim(0., 60.)
    pd.plotfile(fig, 'acorr')
    return fig, ax

def sim_lc(n, mean, n_sin, a_sin):
    x = np.arange(n, dtype = np.float)
    #sampel more or less evenly in frequency
    frq = np.random.rand(n_sin)/2.
    lc = np.ones(n, dtype= np.float)
    for period in 1./frq:
        lc += a_sin * np.sin(x/(period/np.pi))
    lc *= mean
    #add poisson noise
    lc = np.clip(lc, 0., np.inf)
    for i in range(n):
        lc[i] = np.random.poisson(lc[i])
    return lc

def sim_ls(n_mc, *args):
    lspower = np.zeros(n_mc)
    FAP = np.zeros(n_mc)
    for i in range(n_mc):
        if np.mod(i, 10) ==0:
            print 'Running MC sim of lc', i, '/',n_mc
        lc1 = sim_lc(*args)
        lc = pyPeriod.TimeSeries(np.arange(0,len(lc1), dtype = float), lc1)
        ls = pyPeriod.LombScargle(lc, ofac=1, hifac=1)
        lspower[i] =  max(ls.power)
        FAP[i] = ls.FAP(max(ls.power))
    ind = np.argsort(lspower)
    lspower = lspower[ind]
    FAP = FAP[ind]
    return lspower, FAP


def print_MC_LS_FAP(filename, nsum = 1, nsim = 1000, n_sin = 100, a_sin = .01):
    cos = CosCounts(filename, nsum = nsum, deadtime = 7.4e-7)
    FAP_max, P_max = cos.LombScargle()
    N = len(cos.hist)
    ampnoise = (N/4./ P_max - 0.5)**(-0.5)
    sigma = np.sqrt(scipy.var(cos.hist))
    print 'highest LS peak: ', P_max
    print 'Amplitude of sin, that would make this peak = noise *', ampnoise
    print 'This means the relativ Amp is = ', ampnoise * sigma / np.mean(cos.hist)
    # The np.mod is required because pyTiming expects even number of points
    simpower, FAP = sim_ls(nsim, len(cos.hist)-np.mod(len(cos.hist),2), np.mean(cos.hist), n_sin, a_sin)
    simpower.sort()
    print 'more than 90%, 95%, 99% of all simulations have an LS power > ', simpower[(1.-0.9)*nsim], simpower[(1.-0.95) * nsim], simpower[(1.-0.99) * nsim]
    print ' that is FAP of ', FAP[(1.-0.9)*nsim], FAP[(1.-0.95) * nsim], FAP[(1.-0.99) * nsim]

    return simpower


if __name__ == "__main__":

    mpl.rcParams['font.size']=16
    mpl.rcParams['ps.fonttype'] =42
    mpl.rcParams['legend.fontsize']='large'

    tnuv, tuva, tuvb, tciv = analyse_all_COS_evts(pd.datapath, do_FAP = False)

    #fig, ax = plot_FAPs(tnuv, tuva, tuvb, tciv)

    #tab = atpy.Table(str(tuva.file[7]), hdu = 1)
    #ypos = tabmax(tab, 'RAWY', bin = 10)
    #tab = tab.where((tab.RAWY > ypos-40) & (tab.RAWY < ypos+40))
    #hist, bins = np.histogram(tab.TIME, bins = np.arange(-.016,max(tab.TIME)+0.01,0.032))

    #fig, ax = plot_periodogram(hist, bins)
    #fig, ax = plot_lc(hist, bins)
    #fig, ax = plot_acorr(hist, bins)

    fig, ax, fig2, ax2 = plot_sigma(tnuv, tuva, tuvb, tciv, multsim = 1)
    simpower = print_MC_LS_FAP(str(tuva.file[7]), nsum = 1, nsim = 1000, n_sin = 100, a_sin = .05)
