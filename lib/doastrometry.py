import numpy as np
from scipy.optimize import minimize
from scipy import linalg as scipylinalg
import pyfits as fits
import sys
import re
import os
# from pylab import *
import pickle

deg2rad = np.pi / 180.
rad2deg = 180. / np.pi

class WCSsol(object):

    CDscale = 1e-5 #0.27 / 60. / 60. / 4.

    def __init__(self, RAdata, DECdata, i, j, CRPIX, CRVAL, CD, PV):
        
        self.RAdata = RAdata
        self.DECdata = DECdata
        self.i = i
        self.j = j
        self.CRPIX1 = CRPIX[0]
        self.CRPIX2 = CRPIX[1]
        self.CRVAL1 = CRVAL[0]
        self.CRVAL2 = CRVAL[1]
        self.CD11 = CD[0, 0]
        self.CD12 = CD[0, 1]
        self.CD21 = CD[1, 0]
        self.CD22 = CD[1, 1]
        self.PV = PV
        
    # ---------------------------------------

    def x(self):
        
        return self.CD11 * (self.i - self.CRPIX1) + self.CD12 * (self.j - self.CRPIX2) # deg 
  
    def y(self):
        
        return self.CD21 * (self.i - self.CRPIX1) + self.CD22 * (self.j - self.CRPIX2) # deg
  
    # ---------------------------------------

    def r(self):
        
        return np.sqrt(self.x()**2 + self.y()**2) # deg

    def xi(self):

        return deg2rad * (self.PV[0, 0] + self.PV[0, 1] * self.x() + self.PV[0, 2] * self.y() + self.PV[0, 3] * self.r() + self.PV[0, 4] * self.x()**2 + self.PV[0, 5] * self.x() * self.y() + self.PV[0, 6] * self.y()**2 + self.PV[0, 7] * self.x()**3 + self.PV[0, 8] * self.x()**2 * self.y() + self.PV[0, 9] * self.x() * self.y()**2 + self.PV[0, 10] * self.y()**3)

    def eta(self):

        return deg2rad * (self.PV[1, 0] + self.PV[1, 1] * self.y() + self.PV[1, 2] * self.x() + self.PV[1, 3] * self.r() + self.PV[1, 4] * self.y()**2 + self.PV[1, 5] * self.y() * self.x() + self.PV[1, 6] * self.x()**2 + self.PV[1, 7] * self.y()**3 + self.PV[1, 8] * self.y()**2 * self.x() + self.PV[1, 9] * self.y() * self.x()**2 + self.PV[1, 10] * self.x()**3) # rad

    # ---------------------------------------

    def fu(self):
        return self.xi() / np.cos(self.CRVAL2 * deg2rad) # rad

    def fv(self):
        return 1. - self.eta() * np.tan(self.CRVAL2 * deg2rad) # rad

    def fw(self):
        return self.eta() + np.tan(self.CRVAL2 * deg2rad) # rad

    def fa(self):
        return np.arctan2(self.fu(), self.fv()) # rad

    def RA(self):
        return self.CRVAL1 + self.fa() * rad2deg # deg

    def DEC(self):
        return np.arctan2(self.fw() * np.cos(self.fa()), self.fv()) * rad2deg # deg

    # -------------------------------------------

    def drdx(self):
        
        return self.x() / self.r()

    def drdy(self):
        
        return self.y() / self.r()

    def dxidx(self):
    
        return deg2rad * (self.PV[0, 1] + self.PV[0, 3] * self.drdx() + self.PV[0, 4] * 2. * self.x() + self.PV[0, 5] * self.y() + self.PV[0, 7] * 3. * self.x()**2 + self.PV[0, 8] * 2. * self.x() * self.y() + self.PV[0, 9] * self.y()**2)
    
    def dxidy(self):
        
        return deg2rad * (self.PV[0, 2] + self.PV[0, 3] * self.drdy() + self.PV[0, 5] * self.x() + self.PV[0, 6] * 2. * self.y() + self.PV[0, 8] * self.x()**2 + self.PV[0, 9] * self.x() * 2. * self.y() + self.PV[0, 10] * 3. * self.y()**2)
    
    def detadx(self):
    
        return deg2rad * (self.PV[1, 2] + self.PV[1, 3] * self.drdx() + self.PV[1, 5] * self.y() + self.PV[1, 6] * 2. * self.x() + self.PV[1, 8] * self.y()**2 + self.PV[1, 9] * self.y() * 2. * self.x() + self.PV[1, 10] * 3. * self.x()**2)
    
    def detady(self):
    
        return deg2rad * (self.PV[1, 1] + self.PV[1, 3] * self.drdy() + self.PV[1, 4] * 2. * self.y() + self.PV[1, 5] * self.x() + self.PV[1, 7] * 3. * self.y()**2 + self.PV[1, 8] * 2. * self.y() * self.x() + self.PV[1, 9] * self.x()**2)

    # -----------------------------------------
    
    def dfudCRVAL2(self):
        return self.xi() * np.tan(self.CRVAL2 * deg2rad) / np.cos(self.CRVAL2 * deg2rad) * deg2rad
    
    def dfudCRPIX1(self):
        return -(self.dxidx() * self.CD11 + self.dxidy() * self.CD21) / np.cos(self.CRVAL2 * deg2rad)
    
    def dfudCRPIX2(self):
        return -(self.dxidx() * self.CD12 + self.dxidy() * self.CD22) / np.cos(self.CRVAL2 * deg2rad)
    
    def dfudCD11(self):
        return (self.dxidx() * (self.i - self.CRPIX1)) / np.cos(self.CRVAL2 * deg2rad)
    
    def dfudCD12(self):
        return (self.dxidx() * (self.j - self.CRPIX2)) / np.cos(self.CRVAL2 * deg2rad)
    
    def dfudCD21(self):
        return (self.dxidy() * (self.i - self.CRPIX1)) / np.cos(self.CRVAL2 * deg2rad)
    
    def dfudCD22(self):
        return (self.dxidy() * (self.j - self.CRPIX2)) / np.cos(self.CRVAL2 * deg2rad)
    
    # ------------------------------------------
    
    def dfvdCRVAL2(self):
        return -self.eta() * (1. + np.tan(self.CRVAL2 * deg2rad)**2) * deg2rad
    
    def dfvdCRPIX1(self):
        return (self.detadx() * self.CD11 + self.detady() * self.CD21) * np.tan(self.CRVAL2 * deg2rad)
    
    def dfvdCRPIX2(self):
        return (self.detadx() * self.CD12 + self.detady() * self.CD22) * np.tan(self.CRVAL2 * deg2rad)
    
    def dfvdCD11(self):
        return -(self.detadx() * (self.i - self.CRPIX1)) * np.tan(self.CRVAL2 * deg2rad)
    
    def dfvdCD12(self):
        return -(self.detadx() * (self.j - self.CRPIX2)) * np.tan(self.CRVAL2 * deg2rad)
    
    def dfvdCD21(self):
        return -(self.detady() * (self.i - self.CRPIX1)) * np.tan(self.CRVAL2 * deg2rad)
    
    def dfvdCD22(self):
        return -(self.detady() * (self.j - self.CRPIX2)) * np.tan(self.CRVAL2 * deg2rad)
    
    # ------------------------------------------
    
    def dfwdCRVAL2(self):
        return (1. + np.tan(self.CRVAL2 * deg2rad)) * deg2rad
    
    def dfwdCRPIX1(self):
        return -(self.detadx() * self.CD11 + self.detady() * self.CD21)
    
    def dfwdCRPIX2(self):
        return -(self.detadx() * self.CD12 + self.detady() * self.CD22)
    
    def dfwdCD11(self):
        return self.detadx() * (self.i - self.CRPIX1)
    
    def dfwdCD12(self):
        return self.detadx() * (self.j - self.CRPIX2)
    
    def dfwdCD21(self):
        return self.detady() * (self.i - self.CRPIX1)
    
    def dfwdCD22(self):
        return self.detady() * (self.j - self.CRPIX2)
    
    # ----------------------------------------
    
    def dRAdCRVAL1(self):
        return 1.
    
    def dRAdCRVAL2(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCRVAL2() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCRVAL2())
    
    def dRAdCRPIX1(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCRPIX1() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCRPIX1())
    
    def dRAdCRPIX2(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCRPIX2() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCRPIX2())
    
    def dRAdCD11(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCD11() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCD11())
    
    def dRAdCD12(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCD12() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCD12())
    
    def dRAdCD21(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCD21() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCD21())
    
    def dRAdCD22(self):
        return rad2deg / (1. + (self.fu() / self.fv())**2) * (self.dfudCD22() / self.fv() - self.fu() / self.fv()**2 * self.dfvdCD22())

    # -----------------------------------

    def dDECdCRVAL1(self):
        return 0.
    
    def dDECdCRVAL2(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCRVAL2() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCRVAL2() / rad2deg / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCRVAL2())
    
    def dDECdCRPIX1(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCRPIX1() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCRPIX1() / rad2deg / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCRPIX1())
    
    def dDECdCRPIX2(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCRPIX2() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCRPIX2() / rad2deg / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCRPIX2())
    
    def dDECdCD11(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCD11() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCD11() / rad2deg / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCD11())
    
    def dDECdCD12(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCD12() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCD12() / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCD12())
    
    def dDECdCD21(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCD21() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCD21() / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCD21())
    
    def dDECdCD22(self):
        return rad2deg / (1. + (self.fw() * np.cos(self.fa()) / self.fv())**2) * \
            (self.dfwdCD22() * np.cos(self.fa()) / self.fv() - \
             self.fw() * np.sin(self.fa()) * self.dRAdCD22() / self.fv() - \
             self.fw() * np.cos(self.fa()) / self.fv()**2 * self.dfvdCD22())

    # ----------------------------------

    def residuals(self):
        return np.array([self.RA() - self.RAdata, self.DEC() - self.DECdata])

    def chi2(self):
        return np.sum(self.residuals()**2)

    def dchi2dCRVAL1(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.dRAdCRVAL1(), self.dDECdCRVAL1()]))

    def dchi2dCRVAL2(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.dRAdCRVAL2(), self.dDECdCRVAL2()]))

    def dchi2dCRPIX1(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.dRAdCRPIX1(), self.dDECdCRPIX1()]))

    def dchi2dCRPIX2(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.dRAdCRPIX2(), self.dDECdCRPIX2()]))

    def dchi2dCD11(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.dRAdCD11(), self.dDECdCD11()]))

    def dchi2dCD12(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.dRAdCD12(), self.dDECdCD12()]))

    def dchi2dCD21(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.dRAdCD21(), self.dDECdCD21()]))

    def dchi2dCD22(self):
        return 2. * np.sum(self.residuals() * np.vstack([self.dRAdCD22(), self.dDECdCD22()]))

    # jacobian of scalar chi2 funcion  --------------------

    # chi2 as a function of CRPIX CRVAL parameters
    def chi2CRpar(self, x):
        saux = WCSsol(self.RAdata, self.DECdata, self.i, self.j, np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([[self.CD11, self.CD12], [self.CD21, self.CD22]]), self.PV)
        chi2 = saux.chi2()
        del saux

        return chi2

    # chi2 as a function of CD parameters
    def chi2CDpar(self, x):
        saux = WCSsol(self.RAdata, self.DECdata, self.i, self.j, np.array([self.CRPIX1, self.CRPIX2]), np.array([self.CRVAL1, self.CRVAL2]), np.array([[x[0], x[1]], [x[2], x[3]]]), self.PV)
        chi2 = saux.chi2()
        del saux

        return chi2

    # chi2 as a function of all parameters
    def chi2par(self, x):
        saux = WCSsol(self.RAdata, self.DECdata, self.i, self.j, np.array([x[0], x[1]]), np.array([x[2], x[3]]), self.CDscale * np.array([[x[4], x[5]], [x[6], x[7]]]), self.PV)
        chi2 = saux.chi2()
        del saux

        return chi2

    # chi2 jacobian w.r.t CR parameters
    def chi2CRjac(self, x):

        saux = WCSsol(self.RAdata, self.DECdata, self.i, self.j, np.array([x[0], x[1]]), np.array([x[2], x[3]]), np.array([[self.CD11, self.CD12], [self.CD21, self.CD22]]), self.PV)
        jac = np.array([saux.dchi2dCRPIX1(), saux.dchi2dCRPIX2(), saux.dchi2dCRVAL1(), saux.dchi2dCRVAL2()])
        del saux

        return jac

    # chi2 jacobian w.r.t CD parameters
    def chi2CDjac(self, x):

        saux = WCSsol(self.RAdata, self.DECdata, self.i, self.j, np.array([self.CRPIX1, self.CRPIX2]), np.array([self.CRVAL1, self.CRVAL2]), np.array([[x[0], x[1]], [x[2], x[3]]]), self.PV)
        jac = np.array([saux.dchi2dCD11(), saux.dchi2dCD12(), saux.dchi2dCD21(), saux.dchi2dCD22()])
        del saux

        return jac

    # chi2 jacobian w.r.t all parameters
    def chi2jac(self, x):

        saux = WCSsol(self.RAdata, self.DECdata, self.i, self.j, np.array([x[0], x[1]]), np.array([x[2], x[3]]), self.CDscale * np.array([[x[4], x[5]], [x[6], x[7]]]), self.PV)
        jac = np.array([saux.dchi2dCRPIX1(), saux.dchi2dCRPIX2(), saux.dchi2dCRVAL1(), saux.dchi2dCRVAL2(), self.CDscale * saux.dchi2dCD11(), self.CDscale * saux.dchi2dCD12(), self.CDscale * saux.dchi2dCD21(), self.CDscale * saux.dchi2dCD22()])
        del saux

        return jac

class starmatch(object):


    def __init__(self, RA, DEC, mag, i, j, flux, r, flag, CRPIX, CRVAL, CD, PV, doplot):
        
        self.RA = RA
        self.DEC = DEC
        self.mag = mag
        self.i = i
        self.j = j
        self.flux = flux
        self.r = r
        self.flag = flag
        self.CRPIX = CRPIX
        self.CRVAL = CRVAL
        self.CD = CD
        self.PV = PV
        self.doplot = doplot

    def match(self):

        print "Matching sets of stars..."
    
        # sort celestial catalogue stars by flux
        idxsorted = np.argsort(self.mag)
        self.mag = self.mag[idxsorted]
        self.RA = self.RA[idxsorted] * 15.
        self.DEC = self.DEC[idxsorted]

        # sort sextractor catalogue by flux
        idxsorted = np.argsort(self.flux)[::-1]
        self.i = self.i[idxsorted]
        self.j = self.j[idxsorted]
        self.flux = self.flux[idxsorted]
        self.r = self.r[idxsorted]
        self.flag = self.flag[idxsorted]

        # select brightest isolated stars from sextractor catalogue
        npix = 100
        nstars = 60
        masksex = (self.flag == 0) & (self.r < np.percentile(self.r, 90)) & (self.flux > np.percentile(self.flux, 90))
        nsex = min(np.sum(masksex), nstars)
        distmin = np.array(map(lambda i, j: np.min(np.sqrt((self.i[masksex][:nsex][(self.i[masksex][:nsex] != i) & (self.j[masksex][:nsex] != j)] - i)**2 + (self.j[masksex][:nsex][(self.i[masksex][:nsex] != i) & (self.j[masksex][:nsex] != j)] - j)**2)), self.i[masksex][:nsex], self.j[masksex][:nsex]))
        masksexdist = (distmin > npix)
        sol = WCSsol(self.i[masksex][:nsex][masksexdist], self.j[masksex][:nsex][masksexdist], self.i[masksex][:nsex][masksexdist], self.j[masksex][:nsex][masksexdist], self.CRPIX, self.CRVAL, self.CD, self.PV)
        (RA, DEC) = (sol.RA(), sol.DEC())

        # select brightest isolated stars from celestial catalogue
        maskRADECcat = (self.RA > min(RA)) & (self.RA < max(RA)) & (self.DEC > min(DEC)) & (self.DEC < max(DEC))
        nRADECcat = min(np.sum(maskRADECcat), nstars)
        distmin = np.array(map(lambda i, j: np.min(np.sqrt((self.RA[maskRADECcat][:nRADECcat][(self.RA[maskRADECcat][:nRADECcat] != i) & (self.DEC[maskRADECcat][:nRADECcat] != j)] - i)**2 + (self.DEC[maskRADECcat][:nRADECcat][(self.RA[maskRADECcat][:nRADECcat] != i) & (self.DEC[maskRADECcat][:nRADECcat] != j)] - j)**2)), self.RA[maskRADECcat][:nRADECcat], self.DEC[maskRADECcat][:nRADECcat]))
        maskRADECcatdist = (distmin > 100 * 0.27 / 60. / 60.)

        # plot both
        if self.doplot:
            try:
                fig, ax = plt.subplots()
                ax.scatter(RA, DEC, alpha = 0.6, marker = 'd', c = distmin[masksexdist], lw = 0, s = 50)
                ax.scatter(self.RA[maskRADECcat][:nRADECcat][maskRADECcatdist], self.DEC[maskRADECcat][:nRADECcat][maskRADECcatdist], alpha = 0.6, s = 100, c = distmin[maskRADECcatdist])
                ax.set_xlim(min(RA), max(RA))
                ax.set_ylim(min(DEC), max(DEC))
                ax.set_title("Bright, isolated stars")
                ax.set_xlabel("RA [deg]")
                ax.set_ylabel("DEC [deg]")
                plt.savefig("%s/WEB/TESTING/fforster/astrometry/%s_%s_%02i_stars.png" % (HiTSpath, field, CCD, refepoch))
            except:
                print "Problem plotting stars..."

        # find matching star indices and compute offsets
        idxmatch = map(lambda x, y: np.argmin((RA - x)**2 + (DEC - y)**2), self.RA[maskRADECcat][:nRADECcat][maskRADECcatdist], self.DEC[maskRADECcat][:nRADECcat][maskRADECcatdist])
        deltaRA = self.RA[maskRADECcat][:nRADECcat][maskRADECcatdist] - RA[idxmatch]
        deltaDEC = self.DEC[maskRADECcat][:nRADECcat][maskRADECcatdist] - DEC[idxmatch]
        
        # find most common offset
        nclose = np.zeros(np.shape(deltaRA))
        for i in range(len(nclose)):
            dist = np.sqrt((deltaRA - deltaRA[i])**2 + (deltaDEC - deltaDEC[i])**2)
            nclose[i] = np.sum(dist < 5. * 0.27 / 60. / 60.)
        dRA = deltaRA[np.argmax(nclose)]
        dDEC = deltaDEC[np.argmax(nclose)]

        # plot offsets
        if self.doplot:
            fig, ax = plt.subplots()
            ax.scatter(deltaRA[nclose > 1], deltaDEC[nclose > 1], c = nclose[nclose > 1], lw = 0)
            ax.scatter(deltaRA, deltaDEC, c = nclose, lw = 0, marker = '*')
            ax.set_xlabel("delta RA [deg]")
            ax.set_ylabel("delta DEC [deg]")
            ax.set_title("Raw offsets")
            plt.savefig("%s/WEB/TESTING/fforster/astrometry/%s_%s_%02i_stars_delta.png" % (HiTSpath, field, CCD, refepoch))
            
        # correct and recalculate distance to closest star in other catalogue
        sol = WCSsol(self.i[masksex][:nsex], self.j[masksex][:nsex], self.i[masksex][:nsex], self.j[masksex][:nsex], self.CRPIX, self.CRVAL, self.CD, self.PV)
        (RA, DEC) = (sol.RA(), sol.DEC())

        # plot corrected positions
        if self.doplot:
            fig, ax = plt.subplots()
            ax.scatter(RA + dRA, DEC + dDEC)
            ax.scatter(self.RA[maskRADECcat][:nRADECcat], self.DEC[maskRADECcat][:nRADECcat], alpha = 0.6, s = 100, c = 'r')
            ax.set_xlabel("RA [deg]")
            ax.set_ylabel("DEC [deg]")
            ax.set_title("Star positions after offset")
            plt.savefig("%s/WEB/TESTING/fforster/astrometry/%s_%s_%02i_stars_corrected.png" % (HiTSpath, field, CCD, refepoch))
      
        # find index of closest match
        idxmatch = map(lambda x, y: np.argmin((RA + dRA - x)**2 + (DEC + dDEC - y)**2), self.RA[maskRADECcat][:nRADECcat], self.DEC[maskRADECcat][:nRADECcat])
        distmatch = np.sqrt((RA[idxmatch] + dRA - self.RA[maskRADECcat][:nRADECcat])**2 + (DEC[idxmatch] + dDEC - self.DEC[maskRADECcat][:nRADECcat])**2)
        maskdist = (distmatch < 5. * 0.27 / 60. / 60.)
        isexmatch = self.i[masksex][:nsex][idxmatch][maskdist]
        jsexmatch = self.j[masksex][:nsex][idxmatch][maskdist]
        
        RAmatch = self.RA[maskRADECcat][:nRADECcat][maskdist]
        DECmatch = self.DEC[maskRADECcat][:nRADECcat][maskdist]
        sol = WCSsol(RAmatch, DECmatch, isexmatch, jsexmatch, self.CRPIX, self.CRVAL, self.CD, self.PV)

        # plot matched stars
        if self.doplot:
            fig, ax = plt.subplots()
            map(lambda x, y, z, a: ax.plot([x, y], [z, a], c = 'k'), sol.RA(), sol.RAdata, sol.DEC(), sol.DECdata)
            ax.set_xlabel("RA [deg]")
            ax.set_ylabel("DEC [deg]")
            soloffset = WCSsol(RAmatch - dRA, DECmatch - dDEC, isexmatch, jsexmatch, self.CRPIX, self.CRVAL, self.CD, self.PV)
            ax.set_title("%i stars, rms: %f [\"] after offset" % (len(sol.RAdata), np.sqrt(soloffset.chi2() / len(sol.RAdata)) * 60. * 60.))
            plt.savefig("%s/WEB/TESTING/fforster/astrometry/%s_%s_%02i_stars_match.png" % (HiTSpath, field, CCD, refepoch))

        print "Pre match chi2: %f (%i stars)" % (sol.chi2(), len(sol.RAdata))
        sol.CRVAL1 += dRA
        sol.CRVAL2 += dDEC
        print "Post match chi2: %f (%i stars)" % (sol.chi2(), len(sol.RAdata))
        return sol


# find arbitrary order (1, 2, or 3) transformation relating two sets of points
def findtransformation(order, x1, y1, x2, y2):
    
    # solve arbitrary order transformation between two coordinate systems
    # find best transformation relating all these points
    # need to write the system of equations (e.g. for cubic order):
    # x' = a1 + b11 x + b12 y + c11 x^2 + c12 x y + c13 y^2 + d11 x^3 + d12 x^2 y + d13 x y^2 + d14 y^3...
    # y' = a2 + b21 x + b22 y + c21 x^2 + c22 x y + c23 y^2 + d21 x^3 + d22 x^2 y + d23 x y^2 + d24 y^3......
    # X' = X beta
    # we use beta = (a1 b11 b12 c11 c12 c13 d11 d12 d13 d14 a2 b21 b22 c21 c22 c23 d21 d22 d23 d24)^T
    # then e.g. for order 3
    # X' = (x1...xn y1...yn)^T, X = ((1 x1 y1 x1*y1 x1^2 y1^2 x1^2*y1 x1*y1^2 x1^3 y1^3 0 0 0 0 0 0 0 0 0 0) ... (1 xn yn xn*yn xn^2 yn^2 xn^2*yn xn*yn^2 xn^3 yn^3 0 0 0 0 0 0 0 0 0) (0 0 0 0 0 0 0 0 0 0 1 x1 y1 x1*y1 x1^2 y1^2 x1^2*y1 x1*y1^2 x1^3 y1^3) ... (0 0 0 0 0 0 0 0 0 0 1 xn yn xn*yn xn^2 yn^2 xn^2*yn xn*yn^2 xn^3 yn^3)
    # the least squares errors is found that beta which is solution of the following linear system
    # (X^T X) beta = (X^T X')
    # below we use the notation X'->Y
    
    if order == 1:
        nptmin = 3
    elif order == 2:
        nptmin = 6
    elif order == 3:
        nptmin = 10
        
    npt = len(x1)
    if npt < nptmin:
        print "\n\nWARNING: Not enough stars to do order %i astrometric solution (%i)...\n\n" % (order, npt)
        sys.exit(15)
    Y = np.zeros(2 * npt)
    Y[0:npt] = x2
    Y[npt: 2 * npt] = y2
    X = np.zeros((2 * npt, 2 * nptmin))
    iterm = 0
    X[0: npt, iterm] = 1.
    iterm = iterm + 1
    X[0: npt, iterm] = x1
    iterm = iterm + 1
    X[0: npt, iterm] = y1
    iterm = iterm + 1
    if order > 1:
        X[0: npt, iterm] = x1 * x1
        iterm = iterm + 1
        X[0: npt, iterm] = x1 * y1
        iterm = iterm + 1
        X[0: npt, iterm] = y1 * y1
        iterm = iterm + 1
    if order > 2:
        X[0: npt, iterm] = x1 * x1 * x1
        iterm = iterm + 1
        X[0: npt, iterm] = x1 * x1 * y1
        iterm = iterm + 1
        X[0: npt, iterm] = x1 * y1 * y1
        iterm = iterm + 1
        X[0: npt, iterm] = y1 * y1 * y1
        iterm = iterm + 1
    for jterm in range(iterm):
        X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
        X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
        X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
        X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
        X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
        X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
        X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
        X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
        X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
        X[npt: 2 * npt, iterm + jterm] = X[0:npt, jterm]
    # solve
    mat = np.dot(X.transpose(), X)
    rhs = np.dot(X.transpose(), Y)
    try:
        print "Solving order %i transformation (npt: %i)..." % (order, npt)
        if order == 1:
            (a1, b11, b12, a2, b21, b22) = scipylinalg.solve(mat, rhs)
            sol_astrometry = np.array([a1, a2, b11, b12, b21, b22])
        elif order == 2:
            (a1, b11, b12, c11, c12, c13, a2, b21, b22, c21, c22, c23) = scipylinalg.solve(mat, rhs)
            sol_astrometry = np.array([a1, a2, b11, b12, b21, b22, c11, c12, c13, c21, c22, c23])
        elif order == 3:
            (a1, b11, b12, c11, c12, c13, d11, d12, d13, d14, a2, b21, b22, c21, c22, c23, d21, d22, d23, d24) = scipylinalg.solve(mat, rhs)
            sol_astrometry = np.array([a1, a2, b11, b12, b21, b22, c11, c12, c13, c21, c22, c23, d11, d12, d13, d14, d21, d22, d23, d24])
    except:
        print "\n\nWARNING: Error solving linear system when matching pixel coordinate systems\n\n"
        sys.exit(16)

    return sol_astrometry
        
if __name__ == "__main__":

    # apply transformation
    def applytransformation(order, x1, y1, sol):
        
        # this is slow, but I prefer fewer bugs than speed at the moment...
        
        x1t = sol[0] + sol[2] * x1 + sol[3] * y1
        y1t = sol[1] + sol[4] * x1 + sol[5] * y1
        if order > 1:
            x1t = x1t + sol[6] * x1 * x1 + sol[7] * x1 * y1 + sol[8] * y1 * y1
            y1t = y1t + sol[9] * x1 * x1 + sol[10] * x1 * y1 + sol[11] * y1 * y1
        if order > 2:
            x1t = x1t + sol[12] * x1 * x1 * x1 + sol[13] * x1 * x1 * y1 + sol[14] * x1 * y1 * y1 + sol[15] * y1 * y1 * y1
            y1t = y1t + sol[16] * x1 * x1 * x1 + sol[17] * x1 * x1 * y1 + sol[18] * x1 * y1 * y1 + sol[19] * y1 * y1 * y1
            
        return (x1t, y1t)

    def doastrometry(field, CCD, refepoch, dofilter):

        filename = "%s/DATA/%s/%s/%s_%s_%02i_image_crblaster.fits" % (HiTSpath, field, CCD, field, CCD, refepoch)

        # extract header information
        header = fits.open(filename)[0].header
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

        # USNO stars
        USNOfile = "%s/SHARED/%s/%s/CALIBRATIONS/USNO_%s_%s_%02i.npy" % (HiTSpath, field, CCD, field, CCD, refepoch)
        USNO = np.load(USNOfile)
        RAUSNO = np.array(USNO[0], dtype = float)
        DECUSNO = np.array(USNO[1], dtype = float)
        magUSNO = np.array(USNO[3], dtype = float)
    
        if dofilter:
            idx = np.random.choice(len(RAUSNO), len(RAUSNO) * 0.234, replace = False)
            RAUSNO = RAUSNO[idx]
            DECUSNO = DECUSNO[idx]
            magUSNO = magUSNO[idx]
    
        # sextractor file
        sexfile = "%s/SHARED/%s/%s/%s_%s_%02i_image_crblaster.fits-catalogue_wtmap_backsize64.dat" % (HiTSpath, field, CCD, field, CCD, refepoch)
        data = np.loadtxt(sexfile).transpose()
        istars = data[1]
        jstars = data[2]
        fluxstars = data[5]
        rstars = data[8]
        fstars = data[9]

        # match sextractor catalogue to USNO catalogue
        solmatch = starmatch(RAUSNO, DECUSNO, magUSNO, istars, jstars, fluxstars, rstars, fstars, CRPIX, CRVAL, CD, PV, False)
        sol = solmatch.match()
    
        # initial solution
        print "chi2:", sol.chi2()
        print "CRPIX2:", sol.CRPIX1, "CRPIX2:", sol.CRPIX2
        print "CRVAL1:", sol.CRVAL1, "CRVAL2:", sol.CRVAL2
        print "CD11:", sol.CD11, "CD12:", sol.CD12, "CD21:", sol.CD21, "CD22:", sol.CD22, "\n\n"
       
        dotest = False
        
        if dotest:
            sol = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, CRPIX, CRVAL, CD, PV)
            delta = sol.CRPIX1 * 1e-8
            sol2 = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, CRPIX + np.array([delta, 0]), CRVAL, CD, PV)
            print "dchi2/dCRPIX1", (sol2.chi2() - sol.chi2()) / delta / sol.dchi2dCRPIX1()
            delta = sol.CRPIX2 * 1e-8
            sol2 = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, CRPIX + np.array([0, delta]), CRVAL, CD, PV)
            print "dchi2/dCRPIX2", (sol2.chi2() - sol.chi2()) / delta / sol.dchi2dCRPIX2()
            
            delta = sol.CRVAL1 * 1e-8
            sol2 = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, CRPIX, CRVAL + np.array([delta, 0]), CD, PV)
            print "dchi2/dCRVAL1", (sol2.chi2() - sol.chi2()) / delta / sol.dchi2dCRVAL1()
            delta = sol.CRVAL2 * 1e-8
            sol2 = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, CRPIX, CRVAL + np.array([0, delta]), CD, PV) 
            print "dchi2/dCRVAL2", (sol2.chi2() - sol.chi2()) / delta / sol.dchi2dCRVAL2()
    
            delta = sol.CD11 * 1e-8
            sol2 = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, CRPIX, CRVAL, CD + np.array([[delta, 0], [0, 0]]), PV)
            print "dchi2/dCD11", (sol2.chi2() - sol.chi2()) / delta / sol.dchi2dCD11()
            delta = sol.CD12 * 1e-8
            sol2 = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, CRPIX, CRVAL, CD + np.array([[0, delta], [0, 0]]), PV)
            print "dchi2/dCD12", (sol2.chi2() - sol.chi2()) / delta / sol.dchi2dCD12()
            delta = sol.CD21 * 1e-8
            sol2 = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, CRPIX, CRVAL, CD + np.array([[0, 0], [delta, 0]]), PV)
            print "dchi2/dCD21", (sol2.chi2() - sol.chi2()) / delta / sol.dchi2dCD21()
            delta = sol.CD22 * 1e-8
            sol2 = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, CRPIX, CRVAL, CD + np.array([[0, 0], [0, delta]]), PV)
            print "dchi2/dCD22", (sol2.chi2() - sol.chi2()) / delta / sol.dchi2dCD22()
    
        # find new solution that minimizes chi2
        sol = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, CRPIX, CRVAL, CD, PV)
        x0 = np.array([CRPIX[0], CRPIX[1], CRVAL[0], CRVAL[1], sol.CD11 / sol.CDscale, sol.CD12 / sol.CDscale, sol.CD21 / sol.CDscale, sol.CD22 / sol.CDscale])
        print "Running minimization routine..."
        gcsol = minimize(sol.chi2par, x0, method = 'L-BFGS-B', jac = sol.chi2jac)#, options = {'disp': True})
        print gcsol, "\n"
        
        if gcsol.success:
            print "%i stars, rms: %f [\"]" % (len(sol.RAdata), np.sqrt(gcsol.fun / len(sol.RAdata)) * 60. * 60.)
            CRPIX = np.array([gcsol.x[0], gcsol.x[1]])
            CRVAL = np.array([gcsol.x[2], gcsol.x[3]])
            CD = sol.CDscale * np.array([[gcsol.x[4], gcsol.x[5]], [gcsol.x[6], gcsol.x[7]]])
            sol = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, CRPIX, CRVAL, CD, PV)
    
            print "chi2:", sol.chi2()
            print "CRPIX2:", sol.CRPIX1, "CRPIX2:", sol.CRPIX2
            print "CRVAL1:", sol.CRVAL1, "CRVAL2:", sol.CRVAL2
            print "CD11:", sol.CD11, "CD12:", sol.CD12, "CD21:", sol.CD21, "CD22:", sol.CD22, "\n\n"

            astrosol = {}
            astrosol["CRPIX"] = np.array([sol.CRPIX1, sol.CRPIX2])
            astrosol["CRVAL"] = np.array([sol.CRVAL1, sol.CRVAL2])
            astrosol["CD"] = np.array([[sol.CD11, sol.CD12], [sol.CD21, sol.CD22]])
            astrosol["PV"] = sol.PV
            pickle.dump(astrosol, open("%s/SHARED/%s/%s/CALIBRATIONS/astrometry_%s_%s_%02i.pkl" % (HiTSpath, field, CCD, field, CCD, refepoch), 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
            
            fig, ax = plt.subplots()
            ax.scatter(sol.residuals()[0] * 60. * 60., sol.residuals()[1] * 60. * 60.)
            ax.set_xlabel("delta RA [\"]")
            ax.set_ylabel("delta DEC [\"]")
            ax.set_title("%i stars, rms: %f [\"] after chi2 minimization" % (len(sol.RAdata), np.sqrt(sol.chi2() / len(sol.RAdata)) * 60. * 60.))
            plt.savefig("%s/WEB/TESTING/fforster/astrometry/%s_%s_%02i_stars_finalsolution.png" % (HiTSpath, field, CCD, refepoch))

            return sol
         
    # Field, CCD and epoch to process
    HiTSpath = "/home/apps/astro"
    fieldprefix = sys.argv[1]
    refepoch = 2
    field = ""
    if fieldprefix == "Blind14A":
        nfields = 40
    elif fieldprefix == "Blind15A":
        nfields = 50
    else:
        field = fieldprefix
        nfields = 1

    for ifield in range(nfields):

        print ifield, fieldprefix
        if field != fieldprefix:
            field = "%s_%02i" % (fieldprefix, ifield + 1)

        for h in ["N", "S"]:
            for nCCD in range(31):
                CCD = "%s%i" % (h, nCCD + 1)
            

                if field != sys.argv[2] or CCD != sys.argv[3]:
                    continue
                print field, CCD

                try:
                    solf = doastrometry(field, CCD, refepoch, True)
                    sol = doastrometry(field, CCD, refepoch, False)
                except:
                    print "-----> WARNING: Cannot process field %s, CCD %s" % (field, CCD)
                    continue

                # open old solution
                matchdir = "%s/SHARED/%s/%s/CALIBRATIONS" % (HiTSpath, field, CCD)
                files = os.listdir(matchdir)
                CRVAL = np.zeros(2)
                CRPIX = np.zeros(2)
                CD = np.zeros((2, 2))

                rmslim = 1e9
                for filei in files:
                    if re.match("matchRADEC_%s_%s_(.*)-%02i.npy" % (field, CCD, refepoch), filei):
                        matchRADEC = np.load("%s/%s" % (matchdir, filei))
                        if matchRADEC[2] < rmslim:
                            rmslim = matchRADEC[2]
                            filesel = filei

                matchRADEC = np.load("%s/%s" % (matchdir, filesel))
                (afluxADUB, e_afluxADUB, rmsdeg, CRVAL[0], CRVAL[1], CRPIX[0], CRPIX[1], CD[0, 0], CD[0, 1], CD[1, 0], CD[1, 1], nPV1, nPV2, ordersol) = matchRADEC[0:14]

                # unpack sol_astrometry_RADEC terms
                nend = 20
                if ordersol == 2:
                    nend = 26
                elif ordersol == 3:
                    nend = 34
                PV = np.zeros((nPV1, nPV2))
                PV = matchRADEC[nend: nend + int(nPV1 * nPV2)].reshape((int(nPV1), int(nPV2)))

                print ordersol
                sol_astrometry_RADEC = matchRADEC[14: nend]
                
                solnew = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, CRPIX, CRVAL, CD, PV)
                (RA, DEC) = applytransformation(ordersol, solnew.RA(), solnew.DEC(), sol_astrometry_RADEC)
                sol_astrometry_RADEC_test = findtransformation(ordersol, solnew.RA(), solnew.DEC(), sol.RAdata, sol.DECdata)
                (RAtest, DECtest) = applytransformation(ordersol, solnew.RA(), solnew.DEC(), sol_astrometry_RADEC_test)

                solnew2 = WCSsol(sol.RAdata, sol.DECdata, sol.i, sol.j, np.array([solf.CRPIX1, solf.CRPIX2]), np.array([solf.CRVAL1, solf.CRVAL2]), np.array([[solf.CD11, solf.CD12], [solf.CD21, solf.CD22]]), solf.PV)
                
                residuals = np.sqrt((sol.RAdata - RAtest)**2 + (sol.DECdata - DECtest)**2) * 60. * 60.
                print np.sum(residuals < 1.5)
                residuals = np.sqrt((sol.RAdata - RA)**2 + (sol.DECdata - DEC)**2) * 60. * 60.
                print np.sum(residuals < 1.5)
                
                fig, ax = plt.subplots()
                
                ax.scatter((sol.RAdata - RAtest) * 60. * 60., (sol.DECdata - DECtest) * 60. * 60., c = 'k', label = "Pol. sol. 33 stars train, 33 stars test", edgecolors = 'none', alpha = 1, s = 50, marker = '*')
                ax.scatter((sol.RAdata - RA) * 60. * 60., (sol.DECdata - DEC) * 60. * 60., c = 'b', label = "Pol. sol. 9 stars train, 33 stars test", edgecolors = 'none', alpha = 0.3, s = 150, marker = 'o')
                ax.scatter((sol.RAdata - sol.RA()) * 60. * 60., (sol.DECdata - sol.DEC()) * 60. * 60., c = 'r', label = "WCS sol. 33 stars train, 33 stars test", edgecolors = 'none', alpha = 0.5, s = 50)
                ax.scatter((sol.RAdata - solnew2.RA()) * 60. * 60., (sol.DECdata - solnew2.DEC()) * 60. * 60., c = 'orange', label = "WCS sol. 9 stars train, 33 stars test", edgecolors = 'none', alpha = 0.5, s = 50)
                ax.legend(framealpha = 0.4, loc = 'upper left')
                ax.set_xlabel("Delta RA [arcsec]")
                ax.set_ylabel("Delta DEC [arcsec]")
                plt.savefig("%s/WEB/TESTING/fforster/astrometry/astrosoltest.png" % HiTSpath)
