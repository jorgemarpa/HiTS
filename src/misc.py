import numpy as np

# nice plotting
# do e.g.  plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(HMSticks))
#          plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(HMSticks))

deg2rad = np.pi / 180.
rad2deg = 180. / np.pi

def HMSticks(hours, pos):
    if hours < 0:
        sign = '-'
    else:
        sign = ''
    hh = int(hours)
    mm = abs(int((hours - hh) * 60.))
    ss = (abs(hours - hh) - mm / 60.) * 3600
    hh = abs(hh)
    return "%s%02i:%02i:%02i" % (sign, hh, mm, ss)

# tangent projection coordinates
def xieta(x, y, PV): # all in degrees
  r = np.sqrt(x**2 + y**2)
  xicomp = PV[0, 0] + PV[0, 1] * x + PV[0, 2] * y + PV[0, 3] * r + PV[0, 4] * x**2 + PV[0, 5] * x * y + PV[0, 6] * y**2 + PV[0, 7] * x**3 + PV[0, 8] * x**2 * y + PV[0, 9] * x * y**2 + PV[0, 10] * y**3
  etacomp = PV[1, 0] + PV[1, 1] * y + PV[1, 2] * x + PV[1, 3] * r + PV[1, 4] * y**2 + PV[1, 5] * y * x + PV[1, 6] * x**2 + PV[1, 7] * y**3 + PV[1, 8] * y**2 * x + PV[1, 9] * y * x**2 + PV[1, 10] * x**3
  return (xicomp, etacomp)


# RA DEC given pixel coordinates
def RADEC(i, j, CD11, CD12, CD21, CD22, CRPIX1, CRPIX2, CRVAL1, CRVAL2, PV):

  # i, j to x, y
  x  = CD11 * (i - CRPIX1) + CD12 * (j - CRPIX2) # deg 
  y = CD21 * (i - CRPIX1) + CD22 * (j - CRPIX2) # deg

  # if no PV, use linear transformation
  if PV is None:
      print "\n\nWARNING: No PV terms found\n\n"
      return ((CRVAL1 + x) / 15., CRVAL2 + y)

  # x, y to xi, eta
  (xi, eta) = xieta(x, y, PV)

  # xi, eta to RA, DEC
  num1 = (xi * deg2rad) / np.cos(CRVAL2 * deg2rad) # rad
  den1 = 1. - (eta * deg2rad) * np.tan(CRVAL2 * deg2rad) # rad
  alphap = np.arctan2(num1, den1) # rad
  RA  = CRVAL1 + alphap * rad2deg # deg
  num2 = (eta * deg2rad + np.tan(CRVAL2 * deg2rad)) * np.cos(alphap) # rad
  DEC = np.arctan2(num2, den1) * rad2deg # deg

  return (RA / 15., DEC) # hr deg


# apply polynomial transformation
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

# Jacobian
def transformation_J(order, x1, y1, sol):

    x1tx = sol[2]
    y1tx = sol[4]
    if order > 1:
        x1tx = x1tx + 2. * sol[6] * x1 + sol[7] * y1
        y1tx = y1tx + 2. * sol[9] * x1 + sol[10] * y1
    if order > 2:
        x1tx = x1tx + 3. * sol[12] * x1 * x1 + 2. * sol[13] * x1 * y1 + sol[14] * y1 * y1
        y1tx = y1tx + 3. * sol[16] * x1 * x1 + 2. * sol[17] * x1 * y1 + sol[18] * y1 * y1

    x1ty = sol[3]
    y1ty = sol[5]
    if order > 1:
        x1ty = x1ty + sol[7] * x1 + 2. * sol[8] * y1
        y1ty = y1ty + sol[10] * x1 + 2. * sol[11] * y1
    if order > 2:
        x1ty = x1ty + sol[13] * x1 * x1 + 2. * sol[14] * x1 * y1 + 3. * sol[15] * y1 * y1
        y1ty = y1ty + sol[17] * x1 * x1 + 2. * sol[18] * x1 * y1 + 3. * sol[19] * y1 * y1

    return np.array([[x1tx, x1ty], [y1tx, y1ty]])

# Inverse transformation
def applyinversetransformation(order, x1t, y1t, sol):
        
    x0 = np.array([x1t, y1t])
    err = 1.
    while err > 0.01:
        x1ti, y1ti = applytransformation(order, x0[0], x0[1], sol)
        dx1, dx2 = np.linalg.solve(transformation_J(order, x0[0], x0[1], sol), np.array([x1t - x1ti, y1t - y1ti]).transpose())
        x0 = x0 + np.array([dx1, dx2])
        err = np.sqrt(dx1**2 + dx2**2)
        
    return x0[0], x0[1]

# Inverse RA DEC transformation
def inverseRADEC(RA, DEC, i0, j0, CD11, CD12, CD21, CD22, CRPIX1, CRPIX2, CRVAL1, CRVAL2, PV):

    #print "Inverse RADEC", RA, DEC

    delta = 0.001

    ij = np.array([i0, j0])
    err = 1.
    while err > 0.001:
        RAi, DECi = RADEC(ij[0], ij[1], CD11, CD12, CD21, CD22, CRPIX1, CRPIX2, CRVAL1, CRVAL2, PV)
        dRAi = (RADEC(ij[0] + delta, ij[1], CD11, CD12, CD21, CD22, CRPIX1, CRPIX2, CRVAL1, CRVAL2, PV)[0] - RAi) / delta
        dRAj = (RADEC(ij[0], ij[1] + delta, CD11, CD12, CD21, CD22, CRPIX1, CRPIX2, CRVAL1, CRVAL2, PV)[0] - RAi) / delta
        dDECi = (RADEC(ij[0] + delta, ij[1], CD11, CD12, CD21, CD22, CRPIX1, CRPIX2, CRVAL1, CRVAL2, PV)[1] - DECi) / delta 
        dDECj = (RADEC(ij[0], ij[1] + delta, CD11, CD12, CD21, CD22, CRPIX1, CRPIX2, CRVAL1, CRVAL2, PV)[1] - DECi) / delta
        jac = np.array([[dRAi, dRAj], [dDECi, dDECj]])
        (di, dj) = np.linalg.solve(jac, np.array([RA - RAi, DEC - DECi]).transpose())
        ij = ij + np.array([di, dj])
        err = np.sqrt(di**2 + dj**2)
        
    return ij[0], ij[1]
