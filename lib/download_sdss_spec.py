import pyfits
import numpy as np
import urllib
import astropy
import astropy.cosmology 
import astropy.units as u
import sdss_reader
import os
import pylab
import sys
from astroquery.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u

if len(sys.argv)<3:
    
    id0=0
    id1=1000

else:
    id0=np.int(sys.argv[1])
    id1=np.int(sys.argv[2])
    


def extinction_correction(xarr1,data1,error1,RA,DEC):
    """ 
    Corrects spectra from galatic reddening. 
    
    Input:
    xarr1: Wavelenght in Angstroms. It MUST be OBSFRAME
    data1: Flux does not matter the units
    error1:FluxError does not matter the units
    RA, DEC: coordinate RA and DEC in degrees. ICRS system.
    
    To properly run this routhine you need to import:
    from astroquery.irsa_dust import IrsaDust
    import astropy.coordinates as coord
    import astropy.units as u
    import numpy as np
    
    Returns:
    xarr,data,error
    
    """
    c = coord.SkyCoord(RA,DEC , unit="deg")
    table = IrsaDust.get_extinction_table(c)
    wlext1=table['LamEff']*1e4
    Aext1=table['A_SFD']
    wlext=np.zeros(len(wlext1)-8) #
    Aext=np.zeros(len(Aext1)-8) #  This is to avoid repetition of close WL in different
    wlext=wlext1[8:] #photometric systems
    Aext=Aext1[8:]
    
    sorted=np.argsort(wlext)
    wlext=wlext[sorted]
    Aext=Aext[sorted]
    Adata=np.interp(np.log10(xarr),np.log10(wlext),np.log10(Aext))
    Adata=10**Adata


    data=10**Adata*data1
    error=10**Adata*error1
    return xarr,data,error


def Flux_to_Luminosity(data,error, redshift,units=1e-17, Ho=70,OmegaM=0.3 ):
    """
    Converts flux to luminosity
    
    data: flux
    error: flux error
    units: 1e-17 erg/(scm2AA) by default. Always in erg/(scm2AA)
    Ho: Hubble constant in km/s
    OmegaM: Normalized matter density wr to critical density
    
    Here we assume a FlatLCDM model: OmegaDE v= 1- OmegaM
    
    Return:
    data,error in erg/(s\AA)
    """
    cos=astropy.cosmology.FlatLambdaCDM(Ho,OmegaM)
    dL=cos.luminosity_distance(redshift)
    dL=dL.to(u.cm).value                       
    #print  data
    data=4*np.pi*data*units*dL**2
    error=4*np.pi*error*units*dL**2
    return data, error


CATALOG='BOSSDR12' # Can also be 'SDSSDR7'
plot=0 #Plot downloaded spectra ?


download=1

    

SDSS_info_dir='../SDSS_data/' #dir related with the spectra that will be downloaded
#SDSS_spec_root='http://das.sdss.org/spectro/1d_26/' #SDSS DR7 . Root of the web page to
#download spectra.




# Objects with CIV emission that will be downloaded
# Organized by Plate MedianJulianDate fibre
#plate,MJD,fibre=np.genfromtxt(SDSS_info_dir+CIVobjs,unpack=1,dtype='str')
if CATALOG=='SDSSDR7':
    redshift_cut=1.79 # Minimum redshift to guarantee SiIV+OIV]1400 coverage. 
    SDSS_spec_root='http://dr12.sdss3.org/sas/dr12/sdss/spectro/redux/26/spectra/' #DR12, for old DR7 spec
    savespec='../spec/' #where to save the downloaded spectra
    if not os.path.exists(savespec):
        os.mkdir(savespec)

    #---------------------------------------------------------------------
    CIVobjs='CIV_PlateMJdFiber.txt' 
    CIVobjs='TN12_MgIIxCIV.dat'

    index, plate, MJD, fibre, logL3000, FWHMMgII, logL1450,  FWHMCIV=np.genfromtxt(SDSS_info_dir+CIVobjs,unpack=1,dtype='str',skip_header=3)
    plate=np.array([str(np.int(np.float(pl))) for pl in plate])
    MJD=np.array([str(np.int(np.float(mj))) for mj in MJD])
    fibre=np.array([str(np.int(np.float(fib))) for fib in fibre])

    CIVsel='CIV_selected.txt' 
    if os.path.exists(CIVsel) and os.path.isfile(CIVsel):
        os.remove(CIVsel)

    #---------------------------------------------------------------------
    fn = SDSS_info_dir + CIVsel
        
    f = open(fn, "w")
    f.write("#plate\tMJD\tfiber\n")
    f.close()




    #---------------------------------------------------------------------
    redshift_info='HewettWild2010redshift.txt'
    zinfo=np.loadtxt(SDSS_info_dir+redshift_info,dtype='str',skiprows=20)
    plz=np.array([ np.int(zinfo[:,8][i]) for i in range(len(zinfo[:,8])) ])
    mjdz=np.array([ np.int(zinfo[:,9][i]) for i in range(len(zinfo[:,9])) ])
    fibz=np.array([ np.int(zinfo[:,10][i]) for i in range(len(zinfo[:,10])) ])
    z=np.array([ np.float(zinfo[:,3][i]) for i in range(len(zinfo[:,3])) ])
    #----------------------------------------------------------------------
if CATALOG=='BOSSDR12':
    redshift_cut=1.7 # Minimum redshift to guarantee SiIV+OIV]1400 coverage. 
    CIVobjs='CIV_selectedBOSS.txt'
    pyfits_hdu = pyfits.open(SDSS_info_dir+'DR12Q.fits') # Full SDSSDR12 QUASAR CATALOG
    # Complete description  and furter information in 
    #http://www.sdss.org/dr12/algorithms/boss-dr12-quasar-catalog/

    QDR12= pyfits_hdu[1].data   #extracting the data
    
    #Selecting redshift between 1.67 to 2.4 to guarantee rest-frame spectral coverage between
    # ~1350 to ~3080AA to cover from SiOIV to MgII. Spectral obs-frame coverage of BOSS 3600 to 10500AA
    zup=2.3
    zlow=1.7
    wherelow=QDR12['Z_VI']>zlow
    whereup=QDR12['Z_VI']<zup

    np.savetxt(SDSS_info_dir+CIVobjs,np.transpose([QDR12['PLATE'][whereup*wherelow],QDR12['MJD'][whereup*wherelow],QDR12['FIBERID'][whereup*wherelow],QDR12['Z_VI'][whereup*wherelow]]),fmt='%10i %10i %10i %10.3f', header='plate      MJD        fiber      redshift')
    
    #Selecting redshift between 1.67 to 2.4 to guarantee rest-frame spectral coverage between
    # ~1350 to ~3080AA to cover from SiOIV to MgII. Spectral obs-frame coverage of BOSS 3600 to 10500AA

    SDSS_spec_root='http://data.sdss3.org/sas/dr12/boss/spectro/redux/v5_7_0/spectra/' # DR12 BOSS
    savespec='../spec/BOSS/' #where to save the downloaded spectra
    if not os.path.exists(savespec):
        os.mkdir(savespec)

    plate, MJD, fibre, redshifts=np.genfromtxt(SDSS_info_dir+CIVobjs,unpack=1,dtype='str',skip_header=1)
    plate=np.array([str(np.int(np.float(pl))) for pl in plate])[id0:id1]
    MJD=np.array([str(np.int(np.float(mj))) for mj in MJD])[id0:id1]
    fibre=np.array([str(np.int(np.float(fib))) for fib in fibre])[id0:id1]
    redshifts=np.array([np.float(red) for red in redshifts])[id0:id1]
    
    CIVsel='CIV_selected_BOSS.txt' 
    if os.path.exists(CIVsel) and os.path.isfile(CIVsel):
        os.remove(CIVsel)

    #---------------------------------------------------------------------
    fn = SDSS_info_dir + CIVsel
        
    f = open(fn, "w")
    f.write("#plate\tMJD\tfiber\n")
    f.close()


    









#Dowloaded from http://mnras.oxfordjournals.org/content/suppl/2013/01/18/j.1365-2966.2010.16648.x.DC1/mnras0408-2302-SD1.txt
# Col.  1: SDSS name
# Col.  2: RA
# Col.  3: DEC
# Col.  4: z
# Col.  5: z_e
# Col.  6: FIRST Detection status 
# Col.  7: Alternate redshift 
# Col.  8: z estimation method code 
# Col.  9: Plate
# Col. 10: MJD
# Col. 11: fibre
#---------------------------------------------------------------------


#for pl,mjd,fib in zip(plate,MJD,fibre):
for index in range(len(plate)):
    pl=plate[index]
    mjd=MJD[index]
    fib=fibre[index]
    
    #------DOWNLOADING SDSS DR12 FILE WITH THE APPROPIATE STRUCTURE---#
    if len(pl)==3:
        pl1='0'+pl
    else:
        pl1=pl

    if len(fib)==2:
        fib1='00'+fib
    elif len(fib)==1:
        fib1='000'+fib
    elif len(fib)==3:
        fib1='0'+fib
    else:
        fib1=fib
        
    print  pl1, mjd, fib1

                 
    #---Cross matching HW2010 redshifts with DR12 
    if CATALOG=='SDSSDR7':
        wp=(plz==np.int(pl1))
        wf=(fibz==np.int(fib1))
        wm=(mjdz==np.int(mjd))

        try:
            redshift=z[wp*wf*wm][0]
        except:
            print 'object does not match with HW2010'
            continue
    #---Cross matching HW2010 redshifts with DR12 
    if CATALOG=='BOSSDR12':
        redshift=redshifts[index]
    
    print redshift
    
    #fileroot= 'spSpec' +'-'+ mjd + '-' + pl1 + '-' + fib1  SDSS DR7
    fileroot= 'spec' +'-' + pl1 + '-' + mjd + '-' + fib1  #SDSS DR12. I am downloading SDSSDR7 from the DR12 webpage. That is why 
    # the file structure is the same.

    filename= fileroot+  '.fits'
    sdss_file=savespec+filename
    if download==1:
    
        #download_site=SDSS_spec_root + pl1 + '/1d/' + filename #SDSS DR7
        download_site=SDSS_spec_root + pl1  + '/'+filename
        urllib.urlretrieve(download_site, filename=sdss_file)    
        try:
            data,error,xarr,hdr=sdss_reader.read_sdss(sdss_file)
            os.remove(sdss_file)
        except:
            os.remove(sdss_file)
            print download_site, 'could not be downloaded'
            continue

    #------DOWNLOADING SDSS DR12 FILE WITH THE APPROPIATE STRUCTURE---#
        
    
    

    

    
    
    
        

    if redshift> redshift_cut-0.00001:
        if download==1:
            #-----Correcting for extinction-----#
            RA=hdr['RA'];DEC=hdr['DEC']
            xarr,data,error=extinction_correction(xarr,data,error,RA,DEC)
            #-----Correcting for extinction-----#
            xarr=xarr/(1.0+redshift)
            data,error=Flux_to_Luminosity(data,error, redshift,units=np.float(hdr['BUNIT'][0:5]) )
            np.savetxt(savespec+fileroot+'.txt',np.transpose([xarr,data,error]),header='Wavelenght AA   Flux    Error in erg/(sAA)')
            if plot==1:
                pylab.figure()
                pylab.plot(xarr,data)
    
        f = open(fn, "a")
        f.write("\n".join(["\t".join([str(q) for q in [pl1, mjd, fib1]])]) )
        #"\n".join(["\t".join([str(q) for q in p])
        
        f.write("\n")
        f.close()
        
    

    #np.savetxt(np.transpose([xarr,data,error]),savespec+fileroot+'.txt')
    #table = pyfits_hdu[0].data
    #pyfits_hdu=pyfits.open(sdss_file)
    #hdr = pyfits_hdu[0]._header
    #x0=hdr['COEFF0']
    #dx=hdr['COEFF1']
    #table = pyfits_hdu[0].data
    #xarr=np.array([ 10**(x0+dx*i)  for i in range(hdr['NAXIS1'])    ])
#---------------------------------------------------------------------
# Total structure of the file:
# SDSS_spec_root + plate[id] + '/' + 'spSpec' +'-'+ MJD[id] + '-' + plate[id] + '-' fibre + '.fit'
# example: http://das.sdss.org/spectro/1d_26/0276/1d/spSpec-51909-0276-006.fit
#          #--------- SDSS_spec_root-------#plate#----------#MJD-plate-fiber

#---------------------------------------------------------------------


#download_string=




