import getopt
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as scipylinalg
from scipy.spatial import cKDTree
from astropy.table import Table
from misc_func_leftraru import *

jorgepath = '/home/jmartinez/HiTS'
astropath = '/home/apps/astro'
sharepath = '/home/apps/astro/home/jmartinez'
thresh = 1.0      # threshold of catalog
minarea = 3      # min area for detection
verbose = True
nx = 2048
ny = 4096
pixscale = 1.

# select N brightest objects well inside the image edges and that are not
# in crowded regions


def select(x, y, z, tolx, toly, xmax, xmin, ymax, ymin, error, N):

    # mask points too close to the edges
    maskout = (np.abs(x - (xmax + xmin) / 2.) <= np.abs(xmax - xmin) / 2.
               - 2. * tolx) & (np.abs(y - (ymax + ymin) / 2.) <=
                               np.abs(ymax - ymin) / 2. - 2. * toly)

    # find indices of brightest objects in descending order
    idxflux = np.argsort(z[maskout])[::-1]

    # select only objects well inside, sorted by flux in descending order
    xsel = x[maskout][idxflux]
    ysel = y[maskout][idxflux]
    zsel = z[maskout][idxflux]

    # remove points in crowded regions
    isolated = np.ones(len(xsel), dtype=bool)
    for i in range(len(xsel)):
        if not isolated[i]:
            continue
        dist = np.sqrt((xsel - xsel[i])**2 + (ysel - ysel[i])**2)
        dist[i] = error
        idxmin = np.argmin(dist)
        if dist[idxmin] < error:
            isolated[i] = False
            isolated[idxmin] = False

    # maximum number requested
    N = min(N, np.sum(isolated))

    return xsel[isolated][0:N], ysel[isolated][0:N]


# find arbitrary order (1, 2, or 3) transformation relating two sets of points
def findtransformation(order, x1, y1, x2, y2):

    # solve arbitrary order transformation between two coordinate systems
    # find best transformation relating all these points
    # need to write the system of equations (e.g. for cubic order):

    if order == 1:
        nptmin = 3
    elif order == 2:
        nptmin = 6
    elif order == 3:
        nptmin = 10

    npt = len(x1)
    if npt < nptmin:
        print ("\n\nWARNING: Not enough stars to do order"
               "%i astrometric solution (%i)...\n\n" % (order, npt))
        return None
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
            (a1, b11, b12, c11, c12, c13, a2, b21, b22,
             c21, c22, c23) = scipylinalg.solve(mat, rhs)
            sol_astrometry = np.array(
                [a1, a2, b11, b12, b21, b22, c11, c12, c13, c21, c22, c23])
        elif order == 3:
            (a1, b11, b12, c11, c12, c13, d11, d12, d13, d14, a2, b21, b22,
             c21, c22, c23, d21, d22, d23, d24) = scipylinalg.solve(mat, rhs)
            sol_astrometry = np.array([a1, a2, b11, b12, b21, b22, c11,
                                       c12, c13, c21, c22, c23, d11, d12,
                                       d13, d14, d21, d22, d23, d24])
    except BaseException:
        print("\n\nWARNING: Error solving linear system when"
              "matching pixel coordinate systems\n\n")
        sys.exit(16)

    return sol_astrometry


# distance between two sets
def xydistance(x1, x2, y1, y2, delta):

    nsources = 0
    total = 0

    for isource in range(len(x2)):

        distx = (x2[isource] - x1)
        disty = (y2[isource] - y1)
        dist = distx * distx + disty * disty
        idx = np.argmin(dist)

        if dist[idx] < delta**2:
            nsources += 1
            total += dist[idx]

    return nsources, total


# function that tries to find first rough astrometric solution by brute force
def roughastro(x1, x2, y1, y2, deltaxmin, deltaxmax,
               deltaymin, deltaymax, delta):

    ibest = 0
    jbest = 0
    nbest = 0

    for i in np.arange(deltaxmin, deltaxmax, delta):

        for j in np.arange(deltaymin, deltaymax, delta):

            (nsources, dist) = xydistance(x1, x2 + i, y1, y2 + j, delta)
            if nsources >= nbest:
                ibest = i
                jbest = j
                nbest = nsources
                # print ibest, jbest, nsources, dist

    return ibest, jbest


# function that matches two sets of stars using constant shift,
# returning selected positions and shifts
# as well as linear transformation with coefficients and rms
def match(N, pixscale, order, x1, y1, z1, e_z1, r1, x2, y2, z2, e_z2,
          r2, tolx, toly, xmin, xmax, ymin, ymax, error1, error2,
          flux1min, flux1max, flux2min, flux2max):

    # ASTROMETRY
    # ---------------------------------------------------

    if e_z1 is None:
        e_z1 = np.ones(len(z1))
    if e_z2 is None:
        e_z2 = np.ones(len(z2))
    if r1 is None:
        r1 = np.ones(len(z1))
    if r2 is None:
        r2 = np.ones(len(z2))

    testastrometry = False
    if testastrometry:
        print "Flux cuts based on expected number of stars:"
        print ("   flux1min %f, flux1max %f, flux2min %f, flux2max %f" %
               (flux1min, flux1max, flux2min, flux2max))

    if verbose:
        print "%i and %i stars before flux cut" % (len(x1), len(x2))

    x1 = x1[(z1 > flux1min) & (z1 < flux1max)]
    y1 = y1[(z1 > flux1min) & (z1 < flux1max)]
    r1 = r1[(z1 > flux1min) & (z1 < flux1max)]
    e_z1 = e_z1[(z1 > flux1min) & (z1 < flux1max)]
    z1 = z1[(z1 > flux1min) & (z1 < flux1max)]

    x2 = x2[(z2 > flux2min) & (z2 < flux2max)]
    y2 = y2[(z2 > flux2min) & (z2 < flux2max)]
    r2 = r2[(z2 > flux2min) & (z2 < flux2max)]
    e_z2 = e_z2[(z2 > flux2min) & (z2 < flux2max)]
    z2 = z2[(z2 > flux2min) & (z2 < flux2max)]

    if verbose:
        print "%i and %i stars after flux cut" % (len(x1), len(x2))

    # select the brightest stars only from each set until having a similar
    # number of elements
    idxz1 = np.argsort(z1)[::-1]
    idxz2 = np.argsort(z2)[::-1]
    if len(z1) > 1.2 * len(z2):
        naux = int(1.2 * len(z2))
        x1 = x1[idxz1[:naux]]
        y1 = y1[idxz1[:naux]]
        r1 = r1[idxz1[:naux]]
        e_z1 = e_z1[idxz1[:naux]]
        z1 = z1[idxz1[:naux]]
    elif len(z2) > 1.2 * len(z1):
        naux = int(1.2 * len(z1))
        x2 = x2[idxz2[:naux]]
        y2 = y2[idxz2[:naux]]
        r2 = r2[idxz2[:naux]]
        e_z2 = e_z2[idxz2[:naux]]
        z2 = z2[idxz2[:naux]]

    if testastrometry:
        print "%i and %i stars after normalization cut" % (len(x1), len(x2))
        fig, ax = plt.subplots()  # nrows = 2, figsize = (21, 14))
        ax.scatter(y1, x1, marker='o', c='r', s=10, alpha=0.5,
                   edgecolors='none')
        ax.scatter(y2, x2, marker='*', c='b', s=10, alpha=0.5,
                   edgecolors='none')
        ax.axvline(30)
        ax.axvline(ny - 30)
        ax.axhline(30)
        ax.axhline(nx - 30)
        ax.set_ylim(0, nx)
        ax.set_xlim(0, ny)
        #ax[1].set_ylim(0, nx)
        #ax[1].set_xlim(0, ny)
        plt.savefig(
            "%s/TESTING/fforster/astrometry/test_%s_0.png" %
            (webdir, pixscale == 1))

    # first select only sources not in crowded regions and far from the edges
    if pixscale == 1.:
        (x1s, y1s) = select(x1, y1, z1, tolx, toly, xmin, xmax,
                            ymin, ymax, max(error1, error2), N)
        (x2s, y2s) = select(x2, y2, z2, tolx / 2., toly / 2., xmin,
                            xmax, ymin, ymax, max(error1, error2), N)
    else:
        if len(x1) > len(x2):
            (x1s, y1s) = select(x1, y1, z1, tolx, toly, xmin, xmax,
                                ymin, ymax, max(error1, error2), N)
            (x2s, y2s) = select(x2, y2, z2, tolx / 2., toly / 2., xmin,
                                xmax, ymin, ymax, error2, len(x1s))
        else:
            (x2s, y2s) = select(x2, y2, z2, tolx, toly, xmin, xmax,
                                ymin, ymax, max(error1, error2), N)
            (x1s, y1s) = select(x1, y1, z1, tolx / 2., toly / 2., xmin,
                                xmax, ymin, ymax, error1, len(x2s))

    if testastrometry:
        print "%i and %i stars selected by select routine" % (len(x1s), len(x2s))
        fig, ax = plt.subplots()
        ax.scatter(y1s, x1s, marker='o', edgecolors='none', c='r',
                   s=10, alpha=0.5)
        ax.scatter(y2s, x2s, marker='*', edgecolors='none', c='b', s=10,
                   alpha=0.5)
        ax.set_title("Selected", fontsize=8)
        ax.axvline(30)
        ax.axvline(ny - 30)
        ax.axhline(30)
        ax.axhline(nx - 30)
        ax.set_ylim(0, nx)
        ax.set_xlim(0, ny)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_1.png" %
                    (webdir, pixscale == 1))

    if len(x1s) == 0 or len(x2s) == 0:
        print "   ---> WARNING, no matching stars found."
        return None

    # Use brute force approach to find first guess

    dorough = True
    if dorough:

        ibest = 0
        jbest = 0

        print "Refining solution to 25 pixels..."
        (ibest,
         jbest) = roughastro(x1, x2, y1, y2, ibest - 500 * pixscale,
                             ibest + 500 * pixscale, jbest - 500 * pixscale,
                             jbest + 500 * pixscale, 25 * pixscale)

        print ibest, jbest

        print "Refining solution to 5 pixels..."
        (ibest,
         jbest) = roughastro(x1, x2, y1, y2, ibest - 25 * pixscale,
                             ibest + 25 * pixscale, jbest - 25 * pixscale,
                             jbest + 25 * pixscale, 5 * pixscale)

        print ibest, jbest

        print "Refining solution to 2 pixels..."
        (ibest, jbest) = roughastro(x1, x2, y1, y2, ibest - 5 * pixscale,
                                    ibest + 5 * pixscale,
                                    jbest - 5 * pixscale,
                                    jbest + 5 * pixscale, 2 * pixscale)

        print ibest, jbest

        deltax = ibest
        deltay = jbest

    else:

        # find median separation in x and y
        sepx = []
        sepy = []
        for i in range(len(x1s)):
            dist = np.sqrt((x1s[i] - x2s)**2 + (y1s[i] - y2s)**2)
            idxmin = np.argmin(dist)
            sepx.append(x1s[i] - x2s[idxmin])
            sepy.append(y1s[i] - y2s[idxmin])

        # find the highest concentration
        nsep = np.zeros(len(sepx))
        for i in range(len(sepx)):
            nsep[i] = np.sum((np.sqrt((sepx - sepx[i])**2 +
                                      (sepy - sepy[i])**2) < 5. * pixscale))
        idxnsep = np.argmax(nsep)
        sepxcomp = sepx[idxnsep]
        sepycomp = sepy[idxnsep]
        # sepxcomp = np.median(sepx[nsep == max(nsep)])
        # sepycomp = np.median(sepy[nsep == max(nsep)])

        if testastrometry:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.scatter(sepy, sepx, marker='.', edgecolors='none', c='b', s=5)
            ax.scatter(sepycomp, sepycomp, marker='o', facecolors='none',
                       s=100)
            ax.set_title("Minimum separations", fontsize=8)
            plt.savefig("%s/TESTING/fforster/astrometry/test_%s_2.png" %
                        (webdir, pixscale == 1))

        # 0th order correction
        deltax = sepx[idxnsep]  # np.median(sepx)
        deltay = sepy[idxnsep]  # np.median(sepy)

    # mask stars on the edges and apply correction
    mask1 = (x1 > xmin + tolx) & (x1 < xmax - tolx) & \
            (y1 > ymin + toly) & (y1 < ymax - toly)
    mask2 = (x2 > xmin + tolx) & (x2 < xmax - tolx) & \
            (y2 > ymin + toly) & (y2 < ymax - toly)
    x1 = x1[mask1]
    y1 = y1[mask1]
    z1 = z1[mask1]
    x2 = x2[mask2]
    y2 = y2[mask2]
    z2 = z2[mask2]
    x2 = x2 + deltax
    y2 = y2 + deltay

    if testastrometry:
        print "Stars after median correction: %i" % len(x2)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(y1, x1, marker='o', edgecolors='none', c='r', s=10,
                   alpha=0.5)
        ax.scatter(y2, x2, marker='*', edgecolors='none', c='b', s=10,
                   alpha=0.5)
        ax.set_title("Median distance correction", fontsize=8)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_3.png" %
                    (webdir, pixscale == 1))

    # select pairs of points that are farther than error / 2 to a source in
    # the second image after median correction
    sel1 = np.zeros(len(x1), dtype=bool)
    sel2 = np.zeros(len(x2), dtype=bool)
    idx1 = []
    idx2 = []
    for i in range(len(x1)):
        if sel1[i]:
            continue
        dist = np.sqrt((x1[i] - x2)**2 + (y1[i] - y2)**2)
        idxmin = np.argmin(dist)
        if sel2[idxmin]:
            continue
        if dist[idxmin] < max(error1, error2) / 2.:
            idx1.append(i)
            idx2.append(idxmin)
            sel1[i] = True
            sel2[idxmin] = True

    if idx1 == []:
        return (None, None)

    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    x1 = x1[idx1]
    y1 = y1[idx1]
    r1 = r1[idx1]
    z1 = z1[idx1]
    e_z1 = e_z1[idx1]
    x2 = x2[idx2]
    y2 = y2[idx2]
    r2 = r2[idx2]
    z2 = z2[idx2]
    e_z2 = e_z2[idx2]

    if testastrometry:
        print "Stars after distance matching: %i" % len(x1)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(y1, x1, marker='o', edgecolors='none', c='r', s=10,
                   alpha=0.5)
        ax.scatter(y2, x2, marker='*', edgecolors='none', c='b', s=10,
                   alpha=0.5)
        ax.set_title("Distance matching", fontsize=8)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_4.png" %
                    (webdir, pixscale == 1))

        fig, ax = plt.subplots()
        ax.scatter(y1 - y2, x1 - x2, marker='o', edgecolors='none', c='r',
                   s=10, alpha=0.5)
        ax.set_title("Differences after matching", fontsize=8)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_5.png" %
                    (webdir, pixscale == 1))

    # find new highest concentration
    nsep = np.zeros(len(x1))
    for i in range(len(x1)):
        nsep[i] = np.sum(
            (np.sqrt((x1[i] - x2[i])**2 + (y1[i] - y2[i])**2) < 5. * pixscale))
    idxnsep = np.argmax(nsep)
    deltasepxcomp = np.median((x1 - x2)[nsep == max(nsep)])
    deltasepycomp = np.median((y1 - y2)[nsep == max(nsep)])

    # select matched sources, removing outliers
    dist = np.sqrt((x1 - x2 - deltasepxcomp)**2 + (y1 - y2 - deltasepycomp)**2)
    distmask = (dist < 2. * pixscale)  # 5 * pixscale
    if order == 1:
        nptmin = 3
    elif order == 2:
        nptmin = 6
    elif order == 3:
        nptmin = 10
    if np.sum(distmask) < nptmin:
        distmask = (dist < 10. * pixscale)

    x1 = x1[distmask]
    y1 = y1[distmask]
    r1 = r1[distmask]
    z1 = z1[distmask]
    e_z1 = e_z1[distmask]
    x2 = x2[distmask] - deltax
    y2 = y2[distmask] - deltay
    r2 = r2[distmask]
    z2 = z2[distmask]
    e_z2 = e_z2[distmask]

    if testastrometry:
        print "Stars after distance filtering: %i" % np.sum(distmask)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(y1, x1, marker='o', edgecolors='none', c='r',
                   s=10, alpha=0.5)
        ax.scatter(y2, x2, marker='*', edgecolors='none', c='b',
                   s=10, alpha=0.5)
        ax.set_title("Positions after matching and filtering", fontsize=8)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_6.png" %
                    (webdir, pixscale == 1))

        fig, ax = plt.subplots()
        ax.scatter(y1 - y2, x1 - x2, marker='.', edgecolors='none', c='b',
                   s=10, alpha=0.5)
        ax.scatter(deltasepycomp + deltay, deltasepxcomp + deltax,
                   marker='o', s=100, facecolors='none')
        ax.set_title("Differences after matching and filtering", fontsize=8)
        plt.savefig("%s/TESTING/fforster/astrometry/test_%s_7.png" %
                    (webdir, pixscale == 1))

    print "Number of star coincidences: ", len(x1), len(y1)
    # find best transformation relating all these points
    sol_astrometry = findtransformation(order, x1, y1, x2, y2)

    # compute new variables and return together with rms
    if sol_astrometry is not None:
        (x1t, y1t) = applytransformation(order, x1, y1, sol_astrometry)

        # root mean squared error
        rms = np.sqrt(np.sum((x1t - x2)**2 + (y1t - y2)**2) / len(x1))

        return (rms, sol_astrometry)
    else:
        return (None, None)


def calculate_aflux(xref, yref, zref, e_zref, x2, y2,
                    z2, e_z2, order, sol_astrometry):

    # Loading all catalogues
    print 'Applying transformation to pix coordinates'
    x2_ = np.zeros_like(x2)
    y2_ = np.zeros_like(y2)
    for k in range(len(x2)):
        x2_[k], y2_[k] = applyinversetransformation(
            order, x2[k], y2[k], sol_astrometry)

    # Creating tree for ref epoch
    ref_XY = np.transpose(np.array((xref, yref)))
    tree_XY = cKDTree(ref_XY)

    # Calculating aflux with reference epoch
    print 'Compearing epoch %s with ref' % (epo)
    XY_cata = np.transpose(np.array((x2_, y2_)))

    print 'Crossmatching with reference epoch'
    superpos_ind = tree_XY.query(XY_cata, k=1, distance_upper_bound=4)[1]
    # dice los obj de single epoch encontrados en stack
    index_filter = (superpos_ind < len(xref))
    index = superpos_ind[index_filter]

    matched_zref = zref[index]              # objetos encontados en ref epoch
    matched_e_zref = e_zref[index]
    # objetos encontados en single epoch
    matched_z2 = z2[index_filter]
    matched_e_z2 = e_z2[index_filter]

    print('Total of matched objects: %i, %i' %
          (len(matched_zref), len(matched_z2)))

    ratioz = matched_zref / matched_z2
    ratiozMAD = np.median(np.abs(np.median(ratioz) - ratioz))
    maskflux = np.isfinite(matched_zref) & np.isfinite(matched_z2) &\
        np.isfinite(matched_e_zref) & np.isfinite(matched_e_z2) &\
        (matched_zref < 5e5) & (matched_z2 < 5e5) & (matched_zref > 500) &\
        (matched_z2 > 500) & (ratioz > np.median(ratioz) - 3. * ratiozMAD) &\
        (ratioz < np.median(ratioz) + 3. * ratiozMAD)

    aflux = np.dot(matched_zref[maskflux],
                   matched_z2[maskflux]) / np.dot(matched_zref[maskflux],
                                                  matched_zref[maskflux])
    e_aflux = np.dot(matched_zref[maskflux],
                     matched_e_z2[maskflux]) / np.dot(matched_zref[maskflux],
                                                      matched_zref[maskflux])

    return aflux, e_aflux


def calculate_aflux_other_filter(xref, yref, zref, e_zref, x2, y2, z2, e_z2,
                                 order, sol_astrometry, order_ref,
                                 sol_astrometry_ref):

    # Loading all catalogues
    print 'Applying transformation to pix coordinates'
    x2_ = np.zeros_like(x2)
    y2_ = np.zeros_like(y2)
    for k in range(len(x2)):
        x2_[k], y2_[k] = applyinversetransformation(
            order, x2[k], y2[k], sol_astrometry)
    xref_ = np.zeros_like(xref)
    yref_ = np.zeros_like(yref)
    for k in range(len(xref)):
        xref_[k], yref_[k] = applyinversetransformation(
            order_ref, xref[k], yref[k], sol_astrometry_ref)

    # Creating tree for ref epoch
    ref_XY = np.transpose(np.array((xref_, yref_)))
    tree_XY = cKDTree(ref_XY)

    # Calculating aflux with reference epoch
    print 'Compearing epoch %s with ref' % (epo)
    XY_cata = np.transpose(np.array((x2_, y2_)))

    print 'Crossmatching with reference epoch'
    superpos_ind = tree_XY.query(XY_cata, k=1, distance_upper_bound=4)[1]
    # dice los obj de single epoch encontrados en stack
    index_filter = (superpos_ind < len(xref))
    index = superpos_ind[index_filter]

    matched_zref = zref[index]              # objetos encontados en ref epoch
    matched_e_zref = e_zref[index]
    # objetos encontados en single epoch
    matched_z2 = z2[index_filter]
    matched_e_z2 = e_z2[index_filter]

    print 'Total of matched objects: %i, %i' % (len(matched_zref), len(matched_z2))

    ratioz = matched_zref / matched_z2
    ratiozMAD = np.median(np.abs(np.median(ratioz) - ratioz))
    maskflux = np.isfinite(matched_zref) & np.isfinite(matched_z2) &\
        np.isfinite(matched_e_zref) & np.isfinite(matched_e_z2) &\
        (matched_zref < 5e5) & (matched_z2 < 5e5) & (matched_zref > 500) &\
        (matched_z2 > 500) & (ratioz > np.median(ratioz) - 3. * ratiozMAD) &\
        (ratioz < np.median(ratioz) + 3. * ratiozMAD)

    aflux = np.dot(matched_zref[maskflux],
                   matched_z2[maskflux]) / np.dot(matched_zref[maskflux],
                                                  matched_zref[maskflux])
    e_aflux = np.dot(matched_zref[maskflux],
                     matched_e_z2[maskflux]) / np.dot(matched_zref[maskflux],
                                                      matched_zref[maskflux])

    return aflux, e_aflux


if __name__ == '__main__':

    field = ''
    CCD = ''

    filter1 = 'g'
    phot_ref = '02'
    astr_ref = '02'
    order = 3
    stack = False

    # read command line option.
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'h:F:C:b:p:a:s:')
    except getopt.GetoptError as err:
        print help
        sys.exit()

    for o, a in optlist:
        if o in ('-h'):
            print help
            sys.exit()
        elif o in ('-F'):
            field = str(a)
        elif o in ('-C'):
            CCD = str(a)
        elif o in ('-b'):
            filter1 = str(a)
        elif o in ('-p'):
            phot_ref = str(a)
        elif o in ('-a'):
            astr_ref = str(a)
        elif o in ('-s'):
            if a == 'True':
                stack = True
            else:
                stack = False
        else:
            continue

    print 'Filter:', filter1
    print '________________________________________________'

    if not os.path.exists("%s/info/%s" % (jorgepath, field)):
        print "Creating field folder"
        os.makedirs("%s/info/%s" % (jorgepath, field))
    if not os.path.exists("%s/info/%s/%s" % (jorgepath, field, CCD)):
        print "Creating CCD folder"
        os.makedirs("%s/info/%s/%s" % (jorgepath, field, CCD))

    # Loading epochs files
    epochs_c2 = np.loadtxt('%s/info/%s/%s_epochs_%s.txt' %
                           (jorgepath, field, field, filter1), dtype=str)
    if filter1 in ['i', 'u'] and epochs_c2.shape == (2,):
        epochs_c2 = epochs_c2.reshape(1, 2)
    print 'shape of epoch variable: ', epochs_c2.shape
    if len(epochs_c2) == 0:
        print('No observation for filter %s in this field %s' %
              (filter1, field))
        sys.exit()

    if filter1 in ['i', 'u'] and field[:8] == 'Blind15A' and len(epochs_c2) > 1:
        print 'photometric reference: ', epochs_c2[0][0]
        phot_ref = epochs_c2[0][0]

    # loading astrometry reference catalog
    cata_file_ar = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_cat.dat"\
        % (jorgepath, field, CCD, field, CCD, astr_ref,
           str(thresh), str(minarea))
    if not os.path.exists(cata_file_ar):
        print 'No catalog file: %s' % (cata_file_ar)
        sys.exit()
    cata_astr_ref = Table.read(cata_file_ar, format='ascii')
    print 'Shape of ref %s catalogue:' % (filter1), len(cata_astr_ref)
    cata_astr_ref = cata_astr_ref[(cata_astr_ref['X_IMAGE'] > 30) &
                                  (cata_astr_ref['X_IMAGE'] < nx - 30) &
                                  (cata_astr_ref['Y_IMAGE'] > 30) &
                                  (cata_astr_ref['Y_IMAGE'] < ny - 30)]
    print 'Shape of ref %s catalogue:' % (filter1), len(cata_astr_ref)

    # loading photometry reference catalog
    if filter1 == 'g':
        cata_phot_ref = cata_astr_ref
    else:
        cata_file_pr = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_cat.dat"\
            % (jorgepath, field, CCD, field, CCD, phot_ref,
               str(thresh), str(minarea))
        if not os.path.exists(cata_file_pr):
            print 'No catalog file: %s' % (cata_file_pr)
            sys.exit()
        cata_phot_ref = Table.read(cata_file_pr, format='ascii')
        cata_phot_ref = cata_phot_ref[(cata_phot_ref['X_IMAGE'] > 30) &
                                      (cata_phot_ref['X_IMAGE'] < nx - 30) &
                                      (cata_phot_ref['Y_IMAGE'] > 30) &
                                      (cata_phot_ref['Y_IMAGE'] < ny - 30)]

    # calculate for al epochs in epoch file
    for epo in epochs_c2[:, 0]:
        if epo == astr_ref:
            continue
        print 'Loading epoch %s...' % (epo)
        cata_file_c2 = "%s/catalogues/%s/%s/%s_%s_%s_image_crblaster_thresh%s_minarea%s_backsize64_cat.dat"\
            % (jorgepath, field, CCD, field, CCD, epo,
               str(thresh), str(minarea))
        if not os.path.exists(cata_file_c2):
            print 'No catalog file: %s' % (cata_file_c2)
            sys.exit()
        cata_c2 = Table.read(cata_file_c2, format='ascii')
        cata_c2 = cata_c2[(cata_c2['X_IMAGE'] > 30) &
                          (cata_c2['X_IMAGE'] < nx - 30) &
                          (cata_c2['Y_IMAGE'] > 30) &
                          (cata_c2['Y_IMAGE'] < ny - 30)]

        print 'Shape of %s catalogue:' % (filter1), len(cata_c2)

        x1_astr = np.array(cata_astr_ref['X_IMAGE'].tolist())
        y1_astr = np.array(cata_astr_ref['Y_IMAGE'].tolist())
        z1_astr = np.array(cata_astr_ref['FLUX_AUTO'].tolist())
        e_z1_astr = np.array(cata_astr_ref['FLUXERR_AUTO'].tolist())
        r1_astr = np.array(cata_astr_ref['FWHM_IMAGE'].tolist())

        x1_phot = np.array(cata_phot_ref['X_IMAGE'].tolist())
        y1_phot = np.array(cata_phot_ref['Y_IMAGE'].tolist())
        z1_phot = np.array(cata_phot_ref['FLUX_AUTO'].tolist())
        e_z1_phot = np.array(cata_phot_ref['FLUXERR_AUTO'].tolist())
        r1_phot = np.array(cata_phot_ref['FWHM_IMAGE'].tolist())

        x2 = np.array(cata_c2['X_IMAGE'].tolist())
        y2 = np.array(cata_c2['Y_IMAGE'].tolist())
        z2 = np.array(cata_c2['FLUX_AUTO'].tolist())
        e_z2 = np.array(cata_c2['FLUXERR_AUTO'].tolist())
        r2 = np.array(cata_c2['FWHM_IMAGE'].tolist())

        rcrowd = 20
        nstarmin = 100.

        print 'Calculating sol astrometry...',
        print epo, ' vs ', astr_ref

        # try to select the 100 brightest stars in each set to start match
        # solution = [rms, sol_astrometry]
        print 100. * (1. - nstarmin / len(z1_astr))
        print 100. * (1. - nstarmin / len(z2))
        try:
            (rms, sol_astrometry) = match(100, pixscale, order, x1_astr,
                                          y1_astr, z1_astr, e_z1_astr,
                                          r1_astr, x2, y2, z2, e_z2, r2,
                                          100, 100, 0, nx, 0, ny, rcrowd,
                                          rcrowd,
                                          np.percentile(z1_astr, 100. *
                                                        (1. - nstarmin /
                                                         len(z1_astr))),
                                          min(1e6, np.percentile(z1_astr,
                                                                 100)),
                                          np.percentile(z2, 100. *
                                                        (1. - nstarmin /
                                                         len(z2))),
                                          min(1e6, np.percentile(z2, 100)))
        except ValueError:
            print 'Fail!'
            print '________________________________________________'
            continue

        if rms is not None:

            name = 'match_%s_%s_%s-%s' % (field, CCD, epo, astr_ref)
            solution = np.hstack([rms, order, sol_astrometry])
            print solution
            print solution.shape
            np.save('%s/info/%s/%s/%s' %
                    (jorgepath, field, CCD, name), solution)

            # calculate aflux
            if len(epochs_c2) == 1:
                print 'Only one epoch for this filter...'
                print 'Skipping aflux...'
                continue
            print '######', epo
            if epo == phot_ref:
                print '####### phot epoch same as ref', phot_ref
                order_phot_ref = order
                sol_astrometry_phot_ref = sol_astrometry
                print '________________________________________________'
                continue
            print 'Calculating aflux...',
            print epo, ' vs ', phot_ref
            if filter1 in ['g']:
                aflux, e_aflux = calculate_aflux(x1_phot, y1_phot,
                                                 z1_phot, e_z1_phot,
                                                 x2, y2, z2, e_z2,
                                                 order, sol_astrometry)
            else:
                aflux, e_aflux = calculate_aflux_other_filter(x1_phot,
                                                              y1_phot,
                                                              z1_phot,
                                                              e_z1_phot,
                                                              x2, y2, z2,
                                                              e_z2, order,
                                                              sol_astrometry,
                                                              order_phot_ref,
                                                              sol_astrometry_phot_ref)
            name = 'aflux_%s_%s_%s-%s' % (field, CCD, epo, phot_ref)
            solution = np.hstack([aflux, e_aflux])
            print solution
            print solution.shape
            np.save('%s/info/%s/%s/%s' % (jorgepath, field, CCD, name),
                    solution)
            print '________________________________________________'
        else:
            print 'No transformation found'
            print '________________________________________________'
            continue

    print 'Done!'
