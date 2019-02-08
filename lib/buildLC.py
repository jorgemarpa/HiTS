import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import re
import time
import pyfits as fits
from scipy.ndimage.interpolation import affine_transform
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcess
from doastrometry2 import WCS, WCSsol, starmatch
import pickle
import argparse
import seaborn as sb
import pandas as pd

sb.set(style="whitegrid", color_codes=True, context="notebook", font_scale=1.4)

datadir = "/home/apps/astro/data/DATA"
sharedir = "/home/apps/astro/data/SHARED"
webdir = "/home/apps/astro/data/WEB"
jorgepath = '/home/jmartinez/HiTS'

# image difference object (does not contain fits files, only file names
# and calibration data)


class sciencediff(object):

    def __init__(self, field, CCD, science, reference):

        self.field = field
        self.CCD = CCD
        self.science = int(science)
        self.reference = int(reference)

    # routine to find files
    def checkfile(self, datadir, field, CCD, name):
        if not os.path.exists("%s/%s/%s/%s" % (datadir, field, CCD, name)):
            print "ERROR, file %s does not exist" % name

    # set convolution and filenames
    def setfitsfilenames(self, datadir, conv1st):

        self.datadir = datadir
        self.conv1st = conv1st
        if conv1st:
            self.convstring = "%02i-%02it" % (self.science, self.reference)
        else:
            self.convstring = "%02it-%02i" % (self.science, self.reference)

        self.diff = "Diff_%s_%s_%s_grid%02i_lanczos2.fits" % (
            self.field, self.CCD, self.convstring, self.reference)
        self.checkfile(self.datadir, self.field, self.CCD, self.diff)

        self.invVAR = "invVAR_%s_%s_%s_grid%02i_lanczos2.fits" % (
            self.field, self.CCD, self.convstring, self.reference)
        self.checkfile(self.datadir, self.field, self.CCD, self.invVAR)

        self.projected = "%s_%s_%02i_image_crblaster_grid%02i_lanczos2.fits"\
                         % (self.field, self.CCD, self.science, self.reference)
        self.checkfile(self.datadir, self.field, self.CCD, self.projected)

        self.unprojected = "%s_%s_%02i_image_crblaster.fits" % (
            self.field, self.CCD, self.science)
        self.checkfile(self.datadir, self.field, self.CCD, self.unprojected)

        self.refimage = "%s_%s_%02i_image_crblaster.fits" % (
            self.field, self.CCD, self.reference)
        self.checkfile(self.datadir, self.field, self.CCD, self.refimage)

        self.original = "%s_%s_%02i_image.fits.fz" % (
            self.field, self.CCD, self.science)
        self.checkfile(self.datadir, self.field, self.CCD, self.original)

    # set convolution kernel
    def setkernel(self, stars1, stars2):

        print "TEST"

    # set calibrations directory and main variables resulting from calibration
    def setcalibrations(self, sharedir):

        self.sharedir = sharedir

        psffile = "%s/%s/%s/CALIBRATIONS/psf_%s_%s_%s_grid%02i_lanczos2.npy" % (
            self.sharedir, self.field, self.CCD, self.field, self.CCD,
            self.convstring, self.reference)
        if os.path.exists(psffile):
            print psffile
            self.psf = np.load(psffile)
        else:
            print "File %s does not exist" % psffile
            sys.exit()

    # do astrometric calibration of reference frame
    def doWCS(self, epoch):

        # filename
        filename = "%s_%s_%02i_image_crblaster.fits" % (
            self.field, self.CCD, epoch)
        self.checkfile(self.datadir, self.field, self.CCD, filename)

        # name of astrometric solution file
        astrosolfile = "%s/%s/%s/CALIBRATIONS/astrometry_%s_%s_%02i.pkl" % (
            self.sharedir, self.field, self.CCD, self.field, self.CCD, epoch)

        # look for existing astrosolfile if astrosol does not exist
        if os.path.exists(astrosolfile):

            astrosol = pickle.load(open(astrosolfile))

        # otherwise compute astrometric solution
        else:

            # get initial solution from reference header
            header = fits.open("%s/%s/%s/%s" % (self.datadir, self.field,
                                                self.CCD, filename))[0].header
            nPV1 = 2
            nPV2 = 11
            PV = np.zeros((nPV1, nPV2))
            for i in range(nPV1):
                for j in range(nPV2):
                    PV[i, j] = float(header["PV%i_%i" % (i + 1, j)])
            CD = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    CD[i, j] = float(header["CD%i_%i" % (i + 1, j + 1)])
            CRPIX = np.zeros(2)
            CRVAL = np.zeros(2)
            for i in range(2):
                CRPIX[i] = float(header["CRPIX%i" % (i + 1)])
                CRVAL[i] = float(header["CRVAL%i" % (i + 1)])

            # get USNO star catalogue
            USNOfile = "%s/%s/%s/CALIBRATIONS/USNO_%s_%s_%02i.npy" % (
                self.sharedir, self.field, self.CCD, self.field,
                self.CCD, self.reference)
            USNO = np.load(USNOfile)
            RAUSNO = np.array(USNO[0], dtype=float)
            DECUSNO = np.array(USNO[1], dtype=float)
            magUSNO = np.array(USNO[3], dtype=float)

            # sextractor file
            sexfile = "%s/%s/%s/%s_%s_%02i_image_crblaster.fits-catalogue_wtmap_backsize64.dat" % (
                self.sharedir, self.field, self.CCD, self.field,
                self.CCD, epoch)
            data = np.loadtxt(sexfile).transpose()
            istars = data[1]
            jstars = data[2]
            fluxstars = data[5]
            rstars = data[8]
            fstars = data[9]

            # match sextractor catalogue to USNO catalogue
            solmatch = starmatch(RAUSNO, DECUSNO, magUSNO, istars, jstars,
                                 fluxstars, rstars, fstars, CRPIX, CRVAL,
                                 CD, PV, False, "/home/apps/astro/data",
                                 self.field, self.CCD, epoch)
            sol = solmatch.match()

            # initial solution
            print "chi2:", sol.chi2()
            print epoch, "CRPIX1:", sol.CRPIX1, "CRPIX2:", sol.CRPIX2
            print epoch, "CRVAL1:", HMS(sol.CRVAL1), "CRVAL2:", HMS(sol.CRVAL2)
            print "CD11:", sol.CD11, "CD12:", sol.CD12, "CD21:", \
                  sol.CD21, "CD22:", sol.CD22, "\n"

            # find new solution that minimizes chi2
            sol = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j,
                         CRPIX, CRVAL, CD, PV)
            x0 = np.array([CRPIX[0], CRPIX[1], CRVAL[0], CRVAL[1],
                           sol.CD11 / sol.CDscale, sol.CD12 / sol.CDscale,
                           sol.CD21 / sol.CDscale, sol.CD22 / sol.CDscale])
            # print "Running minimization routine to find astometric
            # solution..."
            # , options = {'disp': True})
            gcsol = minimize(sol.chi2par, x0, method='L-BFGS-B',
                             jac=sol.chi2jac)
            # print gcsol, "\n"

            if gcsol.success:
                # print "%i stars, rms: %f [\"]" % (len(sol.RAdata),
                # np.sqrt(gcsol.fun / len(sol.RAdata)) * 60. * 60.)
                CRPIX = np.array([gcsol.x[0], gcsol.x[1]])
                CRVAL = np.array([gcsol.x[2], gcsol.x[3]])
                CD = sol.CDscale * \
                    np.array([[gcsol.x[4], gcsol.x[5]],
                              [gcsol.x[6], gcsol.x[7]]])
                sol = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j,
                             CRPIX, CRVAL, CD, PV)

                print "chi2:", sol.chi2()
                print epoch, "CRPIX2:", sol.CRPIX1, "CRPIX2:", sol.CRPIX2
                print epoch, "CRVAL1:", HMS(sol.CRVAL1), "CRVAL2:",\
                      HMS(sol.CRVAL2)
                print "CD11:", sol.CD11, "CD12:", sol.CD12, "CD21:",\
                      sol.CD21, "CD22:", sol.CD22, "\n"

                astrosol = {}
                astrosol["CRPIX"] = np.array([sol.CRPIX1, sol.CRPIX2])
                astrosol["CRVAL"] = np.array([sol.CRVAL1, sol.CRVAL2])
                astrosol["CD"] = np.array(
                    [[sol.CD11, sol.CD12], [sol.CD21, sol.CD22]])
                astrosol["PV"] = sol.PV
                pickle.dump(
                    astrosol,
                    open("%s/%s/%s/CALIBRATIONS/astrometry_%s_%s_%02i.pkl" %
                         (self.sharedir, self.field, self.CCD,
                          self.field, self.CCD, epoch), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

        # build WCS of reference or science
        WCSobj = WCS(astrosol["CRPIX"], astrosol["CRVAL"],
                     astrosol["CD"], astrosol["PV"])
        if epoch == self.reference:
            self.WCSref = WCSobj
        elif epoch == self.science:
            self.WCSsci = WCSobj

    # get stamp around position
    def getstamp(self, mode, ipix, jpix, deltapix):

        if mode == 'refimage':
            image = fits.open("%s/%s/%s/%s" % (self.datadir, self.field,
                                               self.CCD, self.refimage))[0]
        elif mode == 'original':
            image = fits.open("%s/%s/%s/%s" % (self.datadir, self.field,
                                               self.CCD, self.original))[0]
        elif mode == 'projected':
            image = fits.open("%s/%s/%s/%s" % (self.datadir, self.field,
                                               self.CCD, self.projected))[0]
        elif mode == 'diff':
            image = fits.open("%s/%s/%s/%s" % (self.datadir, self.field,
                                               self.CCD, self.diff))[0]
        elif mode == 'invVAR':
            image = fits.open("%s/%s/%s/%s" % (self.datadir, self.field,
                                               self.CCD, self.invVAR))[0]

        header = image.header
        (ny, nx) = (header["NAXIS1"], header["NAXIS2"])
        self.ny = ny
        self.nx = nx
        if mode == "projected" or mode == "original":
            self.MJD = float(header["MJD-OBS"])
            self.airmass = float(header["AIRMASS"])
            self.exptime = float(header["EXPTIME"])
            self.filtername = header["FILTER"][0]
        elif mode == "refimage":
            self.MJDref = float(header["MJD-OBS"])
            self.airmassref = float(header["AIRMASS"])
            self.exptimeref = float(header["EXPTIME"])
            self.filternameref = header["FILTER"][0]

        return image.data[max(0, int(ipix - deltapix)):
                          min(nx, int(ipix + deltapix + 1)),
                          max(0, int(jpix - deltapix)):
                          min(ny, int(jpix + deltapix + 1))]


# offset image using Spline interpolation
def offset(image, xoff, yoff):

    return affine_transform(image, np.array(
        [[1, 0], [0, 1]]), offset=(xoff, yoff))

# norm of difference between two images


def imagediff(xoff, imageref, image):

    scale = np.sum(image) / np.sum(imageref)

    return np.sum((scale * imageref.flatten() -
                   offset(image, xoff[0], xoff[1]).flatten())**2)


# for contour plotting and for symmetry index
npsf = 21
X, Y = np.meshgrid(np.array(range(npsf)), np.array(range(npsf)))
rs2D = np.array(np.sqrt((X - (npsf - 1.) / 2.)**2 +
                        (Y - (npsf - 1.) / 2.)**2)).flatten()

# do optimal photometry given 1D or 2D psf, flux, var and 1D mask


def getoptphot(psf, flux, invvar, mask):

    aux = (psf.flatten())[mask] * (np.abs(invvar.flatten()))[mask]
    optf = np.sum(aux * (flux.flatten())[mask])
    var_optf = np.sum(aux * (psf.flatten())[mask])
    optf = optf / var_optf
    var_optf = 1. / var_optf

    return optf, np.sqrt(var_optf)

# nice formatting for degrees


def HMS(deg):
    pos = True
    if deg < 0:
        pos = False
        deg = abs(deg)
    H = int(deg)
    M = int((deg - H) * 60.)
    S = (deg - H - M / 60.) * 3600.
    if pos:
        return "%02i:%02i:%05.2f" % (H, M, S)
    else:
        return "-%02i:%02i:%05.2f" % (H, M, S)


# open CCD numbers file
hitsdir = "/home/apps/astro/HiTS"
CCDn = {}
(CCDstring, CCDnumber) = np.loadtxt("%s/etc/CCDnumbers.dat" % hitsdir,
                                    dtype=str).transpose()
CCDnumber = np.array(CCDnumber, dtype=int)
for i in range(len(CCDstring)):
    CCDn[CCDstring[i]] = CCDnumber[i]

# function to convert fluxes into magnitudes given fluxes and errors in
# ADU, the CCD number, the exposure time and the airmass of the
# observation


def ADU2mag(flux, e_flux, CCD, exptime, airmass, azero, kzero):
    mag = np.ones(np.shape(flux)) * 30
    mag_1 = np.ones(np.shape(flux)) * 30
    mag_2 = np.ones(np.shape(flux)) * 30
    fluxp = flux + e_flux
    fluxm = flux - e_flux
    mflux = (flux > 0)
    mfluxp = (fluxp > 0)
    mfluxm = (fluxm > 0)
    mag[mflux] = np.array(-2.5 * np.log10(flux[mflux]) + 2.5 * np.log10(
        exptime) - azero[CCDn[CCD] - 1] - kzero[CCDn[CCD] - 1] * airmass)
    mag_1[mfluxp] = np.array(-2.5 * np.log10(fluxp[mfluxp]) + 2.5 * np.log10(
        exptime) - azero[CCDn[CCD] - 1] - kzero[CCDn[CCD] - 1] * airmass)
    mag_2[mfluxm] = np.array(-2.5 * np.log10(fluxm[mfluxm]) + 2.5 * np.log10(
        exptime) - azero[CCDn[CCD] - 1] - kzero[CCDn[CCD] - 1] * airmass)
    return (mag, mag - mag_1, mag_2 - mag)


def build_LC(name='', field='', CCD='', reference='', mode='',
             ipix=None, jpix=None, RA=None, DEC=None, flux_cat=0,
             eflux_cat=0, plotstamps=False):

    if not os.path.exists("%s/lightcurves/galaxy/%s" %
                          (jorgepath, field)):
        os.makedirs("%s/lightcurves/galaxy/%s" % (jorgepath, field))

    # open difference images
    files = os.listdir("%s/%s/%s" % (datadir, field, CCD))

    scidiffs = []

    for i in sorted(files):

        if re.match("Diff_%s_%s_.*_grid%02i_lanczos2.fits" %
                    (field, CCD, reference), i):

            string = re.findall(
                "Diff_%s_%s_(.*)_grid%02i_lanczos2.fits" %
                (field, CCD, reference), i)[0]

            if re.match("\d\d.*-\d\d.*", string):

                (sci, convsci, ref, convref) = re.findall(
                    "(\d\d)(.*)-(\d\d)(.*)", string)[0]
                conv1st = convref == "t"

                scidiff = sciencediff(field, CCD, int(sci), int(ref))
                scidiff.setfitsfilenames(datadir, conv1st)
                scidiff.setcalibrations(sharedir)
                scidiffs.append(scidiff)

    # stamp size
    npsf = 21
    deltapix = (npsf - 1) / 2  # 10

    # array with MJDs
    MJDs = []
    airmasssci = []
    fluxOK = []

    if scidiffs == []:
        print "No available images with reference %i for field %s, CCD %s" % \
              (reference, field, CCD)
        sys.exit()

    # loop among image diference objects to get stamps and fill MJDs
    for i in scidiffs:

        # print file names and epochs
        print i.projected, i.science, i.reference

        # do WCS sol and print RA, DEC
        i.doWCS(i.reference)
        i.doWCS(i.science)

        # r band
        if i.reference != 2 and (mode == "ij" and "ipix" in locals()):
            print "Reading coordinates from g band"
            mode = "RADEC"
            gcoords = np.load(
                "%s/TESTING/jmartinez/LCs/%s/%s/%s_%s_g_coords.npy" %
                (webdir, field, CCD, field, CCD))
            RA = gcoords["RA"]
            DEC = gcoords["DEC"]
            del ipix, jpix  # important for following if
            print mode, RA, DEC

        # get pixel if input RA, DEC
        if "RA" in locals() and mode == "RADEC":

            (jpix, ipix) = i.WCSref.ij(RA, DEC, 1000, 2000)
            ipix = int(ipix)
            jpix = int(jpix)
            print "New input coordinates:", ipix, jpix

        # get stamps
        if not 'refstamp' in locals():
            refstamp = i.getstamp("refimage", ipix, jpix, deltapix)
            MJDref = i.MJDref
            airmass = i.airmassref
            exptime = i.exptimeref
            filtername = i.filternameref
        else:
            i.MJDref = MJDref
            i.airmassref = airmass
            i.exptimeref = exptime

        if mode != "RADEC":
            (RA, DEC) = i.WCSref.RADEC(jpix, ipix)

        if not 'projstamp' in locals():
            projstamp = i.getstamp("projected", ipix, jpix, deltapix)
        else:
            projstamp = np.dstack(
                [projstamp, i.getstamp("projected", ipix, jpix, deltapix)])

        # check that image is not too close to the edge of the detector
        (jj, ii) = i.WCSsci.ij(RA, DEC, 1000, 2000)
        if jj < 50 or jj > i.ny - 50 or ii < 50 or ii > i.nx - 50:
            fluxOK.append(False)
            print "WARNING, candidate very close to the edge in this observation (%i, %i) of (%i, %i)" % (jj, ii, i.ny, i.nx)
        else:
            fluxOK.append(True)

        if not 'origstamp' in locals():
            origstamp = i.getstamp("original", ii, jj, deltapix)
        else:
            origstamp = np.dstack([origstamp,
                                   i.getstamp("original", ii, jj, deltapix)])

        MJDs.append(i.MJD)
        airmasssci.append(i.airmass)

        if not 'diffstamp' in locals():
            diffstamp = i.getstamp("diff", ipix, jpix, deltapix)
        else:
            diffstamp = np.dstack(
                [diffstamp, i.getstamp("diff", ipix, jpix, deltapix)])

        if not 'invVARstamp' in locals():
            invVARstamp = i.getstamp("invVAR", ipix, jpix, deltapix)
        else:
            invVARstamp = np.dstack(
                [invVARstamp, i.getstamp("invVAR", ipix, jpix, deltapix)])

        if not 'psf' in locals():
            psf = i.psf
        else:
            psf = np.dstack([psf, i.psf])

    print np.shape(psf)
    # replace nans with zeros
    psf[np.invert(np.isfinite(psf))] = 0
    diffstamp[np.invert(np.isfinite(diffstamp))] = 0
    invVARstamp[np.invert(np.isfinite(invVARstamp))] = 0

    # plot averages and find best psf offset
    print "Plotting averages..."

    # find best offset
    diffavg = np.sum(diffstamp, axis=2)
    psfavg = np.sum(psf, axis=2)
    print "Finding best offset with psf..."
    print np.shape(diffstamp), np.shape(diffavg), np.shape(psfavg)
    sol = minimize(imagediff, args=(diffavg, psfavg), x0=[0.5, 0.5],
                   method="Nelder-Mead")
    if sol.success:
        (xoff, yoff) = (sol.x[0], sol.x[1])
        if np.sqrt(xoff**2 + yoff**2) > 4:
            print "----> WARNING: too large offset (%4.1f pixels), setting to zero (look at images)" % np.sqrt(xoff**2 + yoff**2)
            xoff = 0
            yoff = 0
        print "dx: %f, dy: %f" % (xoff, yoff)
    else:
        print sol
        xoff = 0
        yoff = 0
        print "Cannot find offset, assuming no offset"

    # note that in this formulae the j and i are switched
    (RA, DEC) = i.WCSref.RADEC(jpix + yoff, ipix + xoff)
    print "Ref coords: %s, %s (%s, %s)" % (HMS(RA / 15.), HMS(DEC), RA, DEC)

    # plot stamps
    if False:
        fig, ax = plt.subplots(ncols=5, figsize=(20, 6))
        ax[0].imshow(refstamp, interpolation="nearest",
                     cmap='gray', origin='lower')
        ax[1].imshow(np.sum(projstamp, axis=2), interpolation="nearest",
                     cmap='gray', origin='lower')
        ax[2].imshow(diffavg, interpolation="nearest", origin='lower',
                     cmap='gray')
        ax[3].imshow(offset(psfavg, xoff, yoff), interpolation="nearest",
                     cmap='gray', origin='lower')
        ax[4].imshow(np.sum(invVARstamp, axis=2), interpolation="nearest",
                     cmap='gray', origin='lower')

        # fix axes
        for i in range(5):
            ax[i].axes.get_xaxis().set_visible(False)
            ax[i].axes.get_yaxis().set_visible(False)
            # 10 - deltapix, 2 * deltapix - (deltapix - 10))
            ax[i].set_xlim(-0.5, npsf - 0.5)
            # 10 - deltapix, 2 * deltapix - (deltapix - 10))
            ax[i].set_ylim(-0.5, npsf - 0.5)
            ax[i].axvline(10, c='gray')
            ax[i].axhline(10, c='gray')

        # separation and plot
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("%s/lightcurves/galaxy/%s/%s_%s_%02i_dif_stack.png" %
                    (jorgepath, field, name, filtername,
                     reference),
                    tight_layout=True, pad_inches=0.01, facecolor='black',
                    bbox_inches='tight')

    # number of differences and flux
    ndiffs = np.size(scidiffs)
    flux = np.zeros(ndiffs)
    e_flux = np.zeros(ndiffs)
    fluxoff = np.zeros(ndiffs)
    e_fluxoff = np.zeros(ndiffs)

    # flux_ref = np.zeros(ndiffs)
    # e_flux_ref = np.zeros(ndiffs)

    MJDs = np.array(MJDs)
    airmasssci = np.array(airmasssci)
    fluxOK = np.array(fluxOK)

    indx_sort = np.argsort(MJDs)
    MJDs = MJDs[indx_sort]
    fluxOK = fluxOK[indx_sort]
    airmasssci = airmasssci[indx_sort]
    projstamp = projstamp[:, :, indx_sort]
    diffstamp = diffstamp[:, :, indx_sort]
    psf = psf[:, :, indx_sort]
    invVARstamp = invVARstamp[:, :, indx_sort]

    # plot stamps
    if plotstamps:
        print "Plotting series (%i differences)" % (ndiffs)
        fig, ax = plt.subplots(ncols=ndiffs, nrows=5,
                               figsize=(1.6 * ndiffs, 8))

    for i in range(ndiffs):

        if plotstamps:
            ax[0, i].imshow(refstamp, interpolation="nearest",
                            cmap='gray', origin='lower')
            ax[1, i].imshow(projstamp[:, :, i], interpolation="nearest",
                            cmap='gray', origin='lower')
            ax[2, i].imshow(diffstamp[:, :, i], interpolation="nearest",
                            cmap='gray', origin='lower',
                            clim=(np.min(diffstamp[:, :, i].flatten()
                                         [rs2D < 10]),
                                  np.max(diffstamp[:, :, i].flatten()
                                         [rs2D < 5])))
            ax[3, i].imshow(offset(psf[:, :, i], xoff, yoff),
                            interpolation="nearest", cmap='gray',
                            origin='lower')
            ax[4, i].imshow(invVARstamp[:, :, i],
                            interpolation="nearest", cmap='gray',
                            origin='lower')
            if fluxOK[i]:
                ax[0, i].text(1, 1, "%8.2f" % MJDref,
                              fontsize=14, color='orange')
                ax[1, i].text(1, 1, "%8.2f" % MJDs[i],
                              fontsize=14, color='orange')
            else:
                ax[0, i].text(1, 1, "%8.2f*" % MJDref,
                              fontsize=14, color='orange')
                ax[1, i].text(1, 1, "%8.2f*" % MJDs[i],
                              fontsize=14, color='orange')
            for j in range(5):
                ax[j, i].axes.get_xaxis().set_visible(False)
                ax[j, i].axes.get_yaxis().set_visible(False)
                # 10 - deltapix, 2 * deltapix - (deltapix - 10))
                ax[j, i].set_xlim(-0.5, npsf - 0.5)
                # 10 - deltapix, 2 * deltapix - (deltapix - 10))
                ax[j, i].set_ylim(-0.5, npsf - 0.5)
                ax[j, i].axvline(10, c='gray')
                ax[j, i].axhline(10, c='gray')

        # print optimal photometry
        mask = (invVARstamp[:, :, i].flatten() > 0) & (rs2D < 6)
        (flux[i], e_flux[i]) = getoptphot(psf[:, :, i],
                                          diffstamp[:, :, i],
                                          invVARstamp[:, :, i], mask)
        (fluxoff[i], e_fluxoff[i]) = getoptphot(offset(psf[:, :, i],
                                                       xoff, yoff),
                                                diffstamp[:, :, i],
                                                invVARstamp[:, :, i], mask)
    #     (flux_ref[i], e_flux_ref[i]) = getoptphot(offset(psf[:, :, i],
    #                                                      xoff, yoff),
    #                                               refstamp,
    #                                               invVARstamp[:, :, i], mask)
    # print 'flux_ref: ', flux_ref
    # print 'Mean(flux_ref): ', np.mean(flux_ref)
    # print 'Median(flux_ref): ', np.median(flux_ref)
    # print 'Std(flux_ref): ', np.std(flux_ref)
    # print '----------------'
    print 'flux_cat: ', flux_cat
    print 'eflux_cat: ', eflux_cat

    if plotstamps:
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("%s/lightcurves/galaxy/%s/%s_%s_%02i_dif_series.png" %
                    (jorgepath, field, name, filtername, reference),
                    tight_layout=True, pad_inches=0.01, facecolor='black',
                    bbox_inches='tight')
        plt.close(fig)

    # sort by MJDs
    # flux = flux[indx_sort]
    # e_flux = e_flux[indx_sort]
    # fluxoff = fluxoff[indx_sort]
    # e_fluxoff = e_fluxoff[indx_sort]

    # remove nans and values outside definition range
    mask = np.isfinite(fluxoff) & fluxOK
    MJDs = MJDs[mask]
    airmasssci = airmasssci[mask]
    flux = flux[mask]
    e_flux = e_flux[mask]
    fluxoff = fluxoff[mask]
    e_fluxoff = e_fluxoff[mask]
    ndiffs = np.sum(mask)

    org_fluxoff = fluxoff
    org_e_fluxoff = e_fluxoff
    fluxoff = fluxoff + flux_cat
    e_fluxoff = np.sqrt(e_fluxoff**2 + eflux_cat**2)

    # open zero point file given the filtername
    (IDzero, filterzero, azero, e_azero,
     bzero, e_bzero, kzero, e_kzero) = np.loadtxt("%s/etc/zeropoints_%s.txt"
                                                  % (hitsdir, filtername),
                                                  dtype=str).transpose()

    toMag = True
    if toMag:
        IDzero = np.array(IDzero, dtype=int)
        azero = np.array(azero, dtype=float)
        e_azero = np.array(e_azero, dtype=float)
        bzero = np.array(bzero, dtype=float)
        e_bzero = np.array(e_bzero, dtype=float)
        kzero = np.array(kzero, dtype=float)
        e_kzero = np.array(e_kzero, dtype=float)

        # magnitudes
        (mags, e1_mags, e2_mags) = ADU2mag(fluxoff, e_fluxoff, CCD,
                                           exptime, airmass, azero, kzero)

        # magnitudes of 3 sigma and 5 sigma detections
        (mag3sigma, mag5sigma, mag1sigma) = ADU2mag(3 * e_fluxoff,
                                                    2 * e_fluxoff,
                                                    CCD, exptime, airmass,
                                                    azero, kzero)

    doGP = False
    if doGP:
        # prediction xs
        x0 = 1.1
        xt = np.log10(MJDs - MJDs[0] + x0)
        X = xt
        X = np.atleast_2d(X).T

        # observations
        y = fluxoff
        dy = e_fluxoff

        # Gaussian process instance
        gp = GaussianProcess(corr='squared_exponential',
                             theta0=1e-1, thetaL=1e-3, thetaU=2.65,
                             nugget=(dy / y)**2, random_start=100)
        gp.fit(X, y)

        # prediction
        x = np.atleast_2d(np.linspace(min(xt), max(xt), 1000)).T
        y_pred, MSE = gp.predict(x, eval_MSE=True)
        sigma = np.sqrt(MSE)

    if True:

        fig, ax = plt.subplots(nrows=3, figsize=(16, 12))

        # plot flux vs epoch
        ax[0].errorbar(1 + np.array(range(ndiffs)), flux, yerr=e_flux, lw=0,
                       elinewidth=1, label="no offset", alpha=0.5, c='gray',
                       marker='.', markersize=15)
        ax[0].errorbar(1 + np.array(range(ndiffs)), org_fluxoff,
                       yerr=org_e_fluxoff, lw=0,
                       elinewidth=1, label="offset", c='b', marker='.',
                       markersize=15)
        ax[0].plot(1 + np.array(range(ndiffs)), 3. * org_e_fluxoff, ls=':',
                   c='gray', label='3 sigma')
        ax[0].plot(1 + np.array(range(ndiffs)), 5. * org_e_fluxoff, ls='--',
                   c='gray', label='5 sigma')
        ax[0].plot(1 + np.array(range(ndiffs)), -3. * org_e_fluxoff, ls=':',
                   c='gray')
        ax[0].plot(1 + np.array(range(ndiffs)), -5. * org_e_fluxoff, ls='--',
                   c='gray')
        ax[0].legend(framealpha=0.5, loc='best', fontsize='x-small')
        ax[0].set_ylabel("flux")
        ax[0].set_xlabel("epochs")

        # plot flux vs MJD
        ax[1].errorbar(MJDs, org_fluxoff, yerr=org_e_fluxoff, lw=0,
                       elinewidth=1, label="no offset", c='gray', marker='.',
                       alpha=0.5, markersize=15)
        ax[1].errorbar(MJDs, flux, yerr=e_flux, lw=0, elinewidth=1,
                       label="offset", c='b', marker='.', markersize=15)
        ax[1].plot(MJDs, 3. * org_e_fluxoff, ls=':',
                   c='gray', label='3 sigma')
        ax[1].plot(MJDs, 5. * org_e_fluxoff, ls='--',
                   c='gray', label='5 sigma')
        ax[1].plot(MJDs, -3. * org_e_fluxoff, ls=':', c='gray')
        ax[1].plot(MJDs, -5. * org_e_fluxoff, ls='--', c='gray')
        ax[1].legend(framealpha=0.5, loc='best', fontsize='x-small')
        ax[1].set_xlabel("MJD")
        ax[1].set_ylabel("flux")

        # plot evolution in magnitudes
        ax[2].errorbar(MJDs - MJDs[0], mags, yerr=[e1_mags, e2_mags],
                       lw=0, elinewidth=1, c='gray',
                       marker='.', alpha=1., markersize=15)
        ax[2].set_ylim(min(27, max(mags + e2_mags + 0.1)),
                       min(mags - e1_mags - 0.1))
        ax[2].set_ylabel(r"$arcsinh(flux)$")
        ax[2].set_xlabel("MJD + %.1f" % (MJDs[0]))

        plt.savefig("%s/lightcurves/galaxy/%s/%s_%s_%02i_dif_lc.png" %
                    (jorgepath, field, name, filtername, reference),
                    tight_layout=True, bbox_inches='tight')
        plt.close(fig)

    output = np.rec.fromarrays((np.ones(len(MJDs)) * MJDref,
                                np.ones(len(MJDs)) * airmass,
                                np.ones(len(MJDs)) * exptime, MJDs, airmasssci,
                                org_fluxoff, org_e_fluxoff,
                                fluxoff, e_fluxoff,
                                mags, e1_mags, e2_mags, mag1sigma, mag3sigma,
                                mag5sigma),
                               dtype=[('MJDref', float), ('airmassref', float),
                                      ('exptimeref', float), ('MJDsci', float),
                                      ('airmasssci', float),
                                      ('ADUref', float), ('e_ADUref', float),
                                      ('ADUref_zp', float),
                                      ('e_ADUref_zp', float),
                                      ('magref', float), ('e1_magref', float),
                                      ('e2_magref', float),
                                      ('mag1sigmaref', float),
                                      ('mag3sigmaref', float),
                                      ('mag5sigmaref', float)])
    print output.shape

    print "%s/lightcurves/galaxy/%s/%s_%s_%02i_dif_data.npy" %\
        (jorgepath, field, name, filtername, reference)
    np.save("%s/lightcurves/galaxy/%s/%s_%s_%02i_dif_data.npy" %
            (jorgepath, field, name, filtername, reference), output)

    if False:
        output = np.rec.fromarrays((RA, DEC),
                                   dtype=[("RA", float), ("DEC", float)])
        np.save("%s/info/diff_LCs/%s_%s_coords.npy" %
                (jorgepath, name, filtername), output)

    plt.close('all')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help="Object name",
                        required=False, default='galaxy', type=str)
    parser.add_argument('-F', '--field', help="HiTS field",
                        required=False, default='Blind15A_01', type=str)
    parser.add_argument('-C', '--ccd', help="HiTS ccd",
                        required=False, default='N1', type=str)
    parser.add_argument('-r', '--reference', help="Reference epoch",
                        required=False, default=2, type=int)
    parser.add_argument('-m', '--mode', help="Coordinate mode",
                        required=False, default='ij', type=str)
    parser.add_argument('-x', '--xcoord', help="x coordinate",
                        required=False, default=0, type=float)
    parser.add_argument('-y', '--ycoord', help="y coordinate",
                        required=False, default=0, type=float)
    parser.add_argument('-b', '--bright', help="flux in counts",
                        required=False, default=0, type=float)
    parser.add_argument('-l', '--list', help="list of objects",
                        required=False, default=False)
    args = parser.parse_args()
    name = args.name
    reference = args.reference
    mode = args.mode

    if not args.list:

        field = args.field
        CCD = args.ccd
        flux_cat = args.bright

        if mode == "ij":
            ipix = args.ycoord
            jpix = args.xcoord

            build_LC(name=name, field=field, CCD=CCD, reference=reference,
                     mode=mode, ipix=ipix, jpix=jpix, flux_cat=flux_cat,
                     plotstamps=True)
        elif mode == "RADEC":
            RA = args.xcoord
            DEC = args.ycoord

            build_LC(name=name, field=field, CCD=CCD, reference=reference,
                     mode=mode, RA=RA, DEC=DEC, flux_cat=flux_cat,
                     plotstamps=True)

    else:
        print args.list
        ID_table = pd.read_csv(args.list, compression='gzip')
        IDs = ID_table.internalID.values
        fail = []
        for kk, id in enumerate(IDs):
            id = id.strip()
            try:
                print kk + 1, id
                field, ccd, col_pix, row_pix = re.findall(
                    r'(\w+\d+\w?\_\d\d?)\_(\w\d+?)\_(\d+)\_(\d+)', id)[0]
                print field, ccd, int(row_pix), int(col_pix)
                ipix = ID_table.iloc[kk]['Y']
                jpix = ID_table.iloc[kk]['X']
                ra = ID_table.iloc[kk]['raMedian']
                dec = ID_table.iloc[kk]['decMedian']
                flux_cat = ID_table.iloc[kk]['gKronFlux']
                eflux_cat = ID_table.iloc[kk]['gKronFluxErr']
                aux_id = ID_table.iloc[kk]['internalID']
                print ipix, jpix, ra, dec, flux_cat
                mode = 'ij'

                build_LC(name=id, field=field, CCD=ccd, reference=reference,
                         mode=mode, ipix=ipix, jpix=jpix, flux_cat=flux_cat,
                         eflux_cat=eflux_cat, plotstamps=True)
            except:
                fail.append(id)
            # if kk == 5: break
            print '-------------------------------------------'

        # np.savetxt('%s/info/galaxy_lc_build_fail_%S.txt' %
        #            (jorgepath, time.strftime('%d%m%Y_%H%M%S')), np.array(fail))
        print 'Number of LC failed: ', len(fail)
        print 'Done!'
