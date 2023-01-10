# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:19:52 2022

@author: kpangalu
"""

# Disk usage or disk space at terminal
# du -sh * | sort -n
# du -sh *

# ***pick without zeros in the data
a = np.array([5.,1.,3.,0.,0.])
c = a[a!=0.]
print(len(c))

(or)
a = np.array([0.,0.,5.,3.,0.])
c = np.nonzero(a)
print(c)
print(np.mean(c))

****see (bgi_slope_clim_boxpltmul.py)
ms1 = msklon[msklon != 0.]	    ;this will give values
ms = np.where(msklon > 0.)[0] 	;this will give index
#***for loop
for i in range(nln):
	for j in range(nlt):
		trmdat = pr[:,j,i]

#***For loop 
for i in range(0,nln):
	for j in range(0,nlt):
		trmdat = pr[:,j,i]

#change directory and create other directories (Object_detection_metrics.py)
os.chdir('/Users/kpangalu/OneDrive - Qualcomm\Desktop/Progms/')
GT_PATH = os.path.join(os.getcwd(), 'input', 'ground-truth')
print(GT_PATH) 


#*****Loading images (boundingbox_gtruth_pred_yoloformat.py)
imgs_dir = "/Users/kpangalu/OneDrive - Qualcomm/Desktop/Progms/Object_metrics/input/images-optional/"

imgs = glob.glob(imgs_dir + "*.jpg") 
imgnames = []
sl = slice(0, -4)
for rimg in imgs:
    #print("Image files: ", rimg[sl])
    fname = rimg[sl]
    splt1 = rimg.split('\\')[1]
    imgnames.append(splt1)
    
###another way
imgs_dir = "/Users/kpangalu/OneDrive - Qualcomm/Desktop/Progms/Object_metrics/input/images-optional/"
for imgs_dir, dirnames, files in os.walk(imgs_dir):
    print(files)
    
#****array different size (Object_detection_metrics.py)
nd = different size in the loop
tp = [0] *nd

# random numbers
xx = list(range(0,11))
(or)
xx = np.arange(0,10)
or
xx = np.arange(0,9,0.1)interpolation
###This program for no duplicates and corresponding index values(global_pr_loc_ghcn_read.py)
data = [1,1,2,3,4,6,5,4]
inds,unq = [],[]
seen = set()
for i, ele in enumerate(data):
	if ele not in seen:
		inds.append(i)
		unq.append(ele)
	seen.add(ele)
print (seen)
print (inds,unq)
print (len(inds))
 (or)
mylist = ["a", "b", "a", "c", "c"]
mylist1=[1,2,3,4,5,5,6,6]
newlist = list(dict.fromkeys(mylist1))
print(newlist) 

for j in range(len(inds)):
	print (j,data[inds[j]])

#string array declaration (global_pr_loc_ghcn_read.py)
year = np.zeros(len(lnes))
#string array
nme = ['' for x in range(len(lnes))]

(or) (cmip56_clt_sesns_diff_8X3_conn.py)
model5 = ['' for x in range(len(ncfiles5))]
for nf in range(0,nfls5):
	model5[nf] = nc.MODEL

####Duplicate
# Duplicate removing
data = [1,1,2,3,4,6,5,4]
lon = (remove(data))

# Duplicates removing another way (global_pr_loc_ghcn_read.py)
lon = list(set(lonn))
name = list(set(nme))
print (len(lon),len(name))
print (name)

# ***float to integer (ind_max_min_dly2mnly_ncdf.py)
print(np.fix(ddays[0]),int(ddays[-1]))

# ***float array to integer array
gindx = [1., 2., 4., 5.] 
gindx = gindx.astype(int) 

# cldcal_clt_mnly_regrid_ncdf.py
cfi = np.zeros((mns,int(jm),int(im)))*np.nan

# ***output file creation (igra_ind_clim_echyr.py)
ofle = dirpath+'IGRA_clim/'+stnnum+'_yrlyclim.dat'
wfid = open(ofle,'w')
print(ofle); pause()

model ="IGRA" (igra_ind_clim_allstns_echyr.py)
syrr = int('%d' %(years[0]))
eyrr = int('%d' %(years[-1]))

# output file creation (global_pr_loc_ghcn_read.py)
wfid =open('/Analysis/TWS/ind_rivers/ghcn_loc_details.txt','w')
wfid.write(' Stn_ID   syear   lat      lon      elev '+ prgname+'\n')
ii=0
for j in range(len(indxs)):
	if(int(year[indxs[j]]) < 2001):
		ii = ii+1
		print (ii,int(idd[indxs[j]]),nme[indxs[j]],lonn[indxs[j]],latt[indxs[j]],year[indxs[j]],elev[indxs[j]])
		wfid.write('%7d %6d %8.3f %8.3f %8.2f\n' %(int(idd[indxs[j]]),int(year[indxs[j]]),float(latt[indxs[j]]),float(lonn[indxs[j]]),float(elev[indxs[j]])))
		;string also
		wfid.write('%7d %25s %6d %8.3f %8.3f %8.2f \n' %(int(stn_id),str(stn),int(yer),float(lat),float(lon),float(ele)))
	
wfid.close()

# reverse latitudes so they go from south to north.
latitudes = data.variables['lat'][::-1]

#print all elements in an array
Print(*data) or display(*data)

# output file creation (AMSR_trend_regression_confidence.py: good one)
wfid =open('/Analysis/Soil_moisture/results/AMSR_confidence_regression2.dat','w')
wfid.write(' Date   lower(95%)   Upper(95%) '+ prgname+'\n')
for i in range(len(p_x)):
	print (i,p_x[i],lower[i],upper[i])
	wfid.write('%7.2f %9.5f %9.5f\n' %(float(p_x[i]),float(lower[i]),float(upper[i])))
wfid.close()

# ***(igra_echstn_blh_seasonavg.py)
outnme = dirpath+'/BLH_season/'+stnnum+'_refblh_season_'+('%4i' %(year1[0]))+'-'+('%4i' %(year1[-1]))+'.txt'
print(outnme)
wfid = open(outnme,'W')
wfid.write(' year   win        spr       sum       fal      yrly '+ prgname+'\n')

#***Writing into a file(python 3) (imd_hist_cross_spec_coherence.py)
ofile = open(os.path.join(dirm+'imd_NorESM1-ME_cross_spec_cohtest.dat'),'w')
prgname1 = os.path.basename(sys.argv[0]); print (prgname1)
ofile.write('   IMD    Hist    freq     tme       psdimd     psdhist   crsspow  coherence\t'+str(prgname)+ '\n')

for k in range(len(tme)):
    # print >> ofile, ('%7.2f %7.2f %7.4f %9.4f %9.4f %9.4f %9.4f %9.4f') %(imd[k],hist[k],f[k],tme[k],dB(fki)[k],dB(fkj)[k],dB(cij)[k],coh[k])
    # ofile.write(('%7.2f %7.2f %7.4f %9.4f %9.4f %9.4f %9.4f %9.4f') %(imd[k],hist[k],f[k],tme[k],dB(fki)[k],dB(fkj)[k],dB(cij)[k],coh[k]))
    ofile.write(('%7.3f %7.3f') %(imd[k],hist[k])+'\n')
    # anofle.write(str(time[i])+'\t'+str(an_avg[i])+'\t'+str(an_detrended[i])+'\n')
ofile.close()

:*************************reading the datasets
fname = '/Analysis/Python/data_mixed.txt'
dinput=np.loadtxt(fname,
	dtype={'names':('Stn_id','name' ,'syear','lat','lon','elev'),
	'formats':(np.int, '|S25',np.int,np.float,np.float,np.float)},
	skiprows=1)
print (dinput.syear)


# ***writing into a netcdf file (era5_zon_tem_pklat_dly.py)
model ="ERA5"
syrr = int('%d' %(stdte))
eyrr = int('%d' %(endte))
print(syrr,eyrr)

outfile = '/Analysis/ERA-Interim/Results/'+ model +"_tem_zon_137lvls_60N-90N_dly_"+ str(syrr)+"-"+str(eyrr)+"_py.nc"
print (outfile)

;*******************pause module
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...") #PY2.7
    programPause = input("Press the <ENTER> key to continue...") #PY3.0

;***********************************************another way to read the text files
#Reading the data file (rf_wet_dry_grouphisto_plt.py)
dirm = '/Analysis/Active_spell/'
rfle = 'rf_wet_dry_days_1901-2016.dat'
rfinput = np.loadtxt(os.path.join(dirm,rfle),skiprows=1)
print ('File size: ',rfinput.shape)
(or) 
import pprint
pprint.pprint(rfinput.size)

dte = rfinput[:,0]
wet = rfinput[:,1]
dry = rfinput[:,2]

Removing the quatation from the string 
import re
a1 = 'samp"le s"tring'
a2 = re.sub('"','',a1)
print(a2)   

;*****printing
a = [1, 2, 3, 4, 5] 
for i in range(len(a)):
	print (a[i]),		#one line
	print (a[i])        #one by one

# - search for station file (bgi_locfiles_pick.py)
stat_f = [os.path.join(ts_dir, x) for x in os.listdir(ts_dir) if stn_nme.strip().replace(' ', '_') in x][0]
print(stat_f)
d_input = np.loadtxt(stat_f, skiprows=1)

;****lat and longitude creation (eof_analysis.py)
lats = [-19.875+x*2.5 for x in range(0, nlat)]
lons = [-154.875+x*2.5 for x in range(0, nlon)]

# *****Picking the particular dates (rf_wet_dry_histo_plt.py)
sdt1, = np.where(dte == 1901)[0] (or) sdt1, = np.nonzero(dte == 1901)[0]
edt1 = np.where(dte == 1980)[0] (or) 2005.9166666666667
print sdt1,edt1,dte[sdt1],dte[edt1]
dte1 = dte[sdt1:edt1]
print dte1

# ***********Picking the periods from 1980 to 2014 (trmm_waves_extract.py)
ind = np.where((d_time >= 1980) & (d_time <= 2015))
d_time = d_time[ind]
nino34 = nino34[ind]

;****read and split the datasets (global_pr_loc_ghcn_read.py or bgi_locfiles_pick.py)
for ln in range(0,10):
	# stn_id = lnes[ln].split(',')[0][:8]
	# stn_nme = lnes[ln].split(',')[0][9:33]
	# syer = lnes[ln].split(',')[0][34:40]
	# latt = lnes[ln].split(',')[0][42:50]
	stn_id = lnes[ln].split('\t')[0][:8]
	stn_nme = lnes[ln].split('\t')[0][9:33]
	syer = int(lnes[ln].split('\t')[0][35:40])
	latt = float(lnes[ln].split('\t')[0][42:49])
	lonn = float(lnes[ln].split('\t')[0][50:58])
	elevv = float(lnes[ln].split('\t')[0][59:68])

**********CSV file reading
Python txt or csv files reading and skipping the rows (cross_corr1.py or global_pr_loc_read.py)
#*** Data location
dir1 = os.path.join('/','Volumes','UCI_ESS','NOAA-GHCN','input.dir','GOSD-Daily')
file1 = 'CDO9213547695576_index.csv'

dinput = os.path.join(dir1,file1)
# dinput = np.loadtxt(os.path.join(dir1,file1),skiprows=1) (or)
# dinput = os.path.join(dir1,'CDO9213547695576_index.csv')

;***reading the text data sets with commas
fid = open(dinput,'r')
lnes = fid.readlines()[1:]
print len(lnes)

;***reading the datasets with commas and skipping the header liens (qbo_multiregrsn.py) 
sfle=dirpath+'SOI_index_1951-2017new.dat'
fid = open(sfle,'r')
lnes = fid.readlines()[3:]
print(len(lnes))
import csv
reader = csv.reader(lnes)
for line in reader:
	# fields = line.split(",")
	field1 = line[1]
	print(field1)

;********output file# output file creation (global_pr_loc_read.py)
wfid =open('/Analysis/test/test.txt','w')
wfid.write(' Stn_ID   syear   lat      lon    elev '+ prgname+'\n')


*****Picking the nearest value from the data (bgi_locfiles_pick.py)
indx = (np.abs(dtes-2002.)).argmin()
print (indx,dtes[indx])

;***picking the particular lon/lat values (This is IDL)
slon = where(lon ge 88.329) & slon=fix(slon(0)) 
slat = where(lat le 26.681) & slat=fix(slat(0))
print,slon,slat,lon(slon),lat(slat)

;****picking the particular Year (rf_wet_dry_test.py)
sdt1, = np.nonzero(dte == 1902)[0]
edt1, = np.where(dte == 1980)[0]
print sdt1,edt1,dte[sdt1],dte[edt1]

;***Finding year from file 
strlen = len((os.path.join(dirpath, ncfiles[0])))
syer = int((os.path.join(dirpath, ncfiles[0])).split('\t')[0][strlen-16:strlen-12])
print (syer)
pause()

# Historical (cmip6_hist_selmdls_global_yrly_ncdf.py)
h6year = np.zeros((154))*np.nan
h6mean = np.zeros((nfls,154))*np.nan 
for nt in range(0,nfls):
	if(nt == 0): ncfile=dirpath+'BCC-CSM2-MR_hist_stem_1x1_yrly_1861-2014.nc'
	if(nt == 1): ncfile=dirpath+'BCC-ESM1_hist_stem_1x1_yrly_1861-2014.nc' 
# **************************************
# *****Year, month and digital months(bgi_locfiles_pick.py)
# *** Years start from perfect ex. (185001 to 200512)
yrr=2001
hyer=np.empty(len(hmons), dtype=int)
hmon=np.empty(mns, dtype=int)
hmonths=np.empty(len(hmons), dtype=float)
for mn in range(mns):
	hmon[mn] = (mn % 12)+1
	if(mn > 0) and (mn % 12 == 0):	yrr=yrr+1 
	hyer[mn] = yrr
	hmonths[mn] = yrr + (mn % 12)/12.
# 	print mn,hyer[mn],hmon[mn],hmonths[mn]

years = syer + np.arange(0,(mns/12))
nyrs = len(years)

#Monthly climatology and normalalization over std
cmonths = np.zeros(nn,dtype=np.int)
for i in range(nn): cmonths[i] = (i %12)

# *****Year, month and degital months(bgi_locfiles_pick.py)
# ***if years start from 185012 to 200512 (cmip5_srftem_sesns_bilinear_syrs_regrid_ncdf.py)
strlen = len((os.path.join(dirpath, ncfiles[0])))
styer = int((os.path.join(dirpath, ncfiles[0])).split('\t')[0][strlen-16:strlen-12])
stmon = int((os.path.join(dirpath, ncfiles[0])).split('\t')[0][strlen-12:strlen-10])
yrr=styer

print('Start year: ',styer,'  ',stmon)
hyer=np.empty(len(tmons), dtype=int)
hmonths = np.empty((mns),dtype=float)
hmon=np.empty(mns, dtype=int)
for mn in range(mns):
	cmn = stmon+mn 
	hmon[mn] = ((cmn-1) % 12)+1
	if(mn > 0) and ((cmn-1) % 12 == 0):	yrr=yrr+1 
	hyer[mn] = yrr
	hmonths[mn] = yrr + ((cmn-1) %12)/12.
	print(mn,hyer[mn],hmon[mn],hmonths[mn])

# removing duplicates
years = list(dict.fromkeys(hyer))
nyrs = len(years)
# *****digital date to year and month(bgi_locfiles_pick.py)
yer=np.empty(len(dtes), dtype=int)
mon=np.empty(len(dtes), dtype=int)
for j in range(len(dtes)):
	ddt = dtes[j]
	yer[j] = math.trunc(ddt)
	mon[j] = int(round(0.5+12*(ddt % math.trunc(ddt))))
	# print j, ddt,yer[j],mon[j]
	# pause()

# ***create month values
mns = len(cmonths)
clmon = np.zeros(mns,dtype=int)
for i in range(mns): clmon[i] = (i % 12)+1

# ***if months starts ex 07 of the year (cldcal_clt_sesns_regrid_ncdf.py)
# ***digital months
hmonths = np.empty((mns),dtype=float)
for mn in range(mns):
	cmn = mons[mn]
	hmonths[mn] = yers[mn] + ((cmn-1) % 12)/12.
	print(mn,cmn,hmonths[mn],yers[mn])


#picking ABL stn corresponding surface station (igra_corr_abl_sur.py)
    statn, = np.nonzero(ablnum == surstnn)[0]
    print(ablnum,statn,surstnn[statn])


*************************** - search for station file(bgi_locfiles_pick.py)

stat_f = [os.path.join(ts_dir, x) for x in os.listdir(ts_dir) if stn_nme.strip().replace(' ', '_') in x][0]
print(stat_f)

;***data reading (sm_clim_normal.py)
dir1 = os.path.join('/','Analysis','Soil_moisture','results')
file1 = 'AMSR_SM_D_mnly_indreg_tmeseries.dat'
dinput1 = np.loadtxt(os.path.join(dir1,file1),skiprows=1)

idx = dinput1[:,0].astype(float)
nn = len(idx)

amsrdte = np.empty(nn)*np.nan
amsrdte = dinput1[:,0]

amsrmnly = np.empty(nn)*np.nan
amsrmnly = dinput1[:,1]
amsrmnly[amsrmnly >10.] = np.nan

# regrids = np.empty((im,jm), dtype=float)
regrids = np.zeros((int(im), int(jm)))*np.nan

************************string split
s = '1234.12'
a,b = map(int,s.split(".",1))
print a, b

ddt = '2000.04235'
aa= ddt.split(".")
print int(aa[0]),round(12./100000.*float(aa[1])+0.5)

;****************************selection of nearest longitude value
slon = 70.683
vall = min(enumerate(hlon), key=lambda x:abs(x[1]-slon))
print vall[0], vall[1]

*****************************Making gap data to in order
****dates modifying index (read_noaa_index.py)
gdates = (np.empty(144, dtype=int))*np.NaN
		gpr = np.empty(144)*np.NaN
		# *****month date index creation
		syr = 2002
		for i in range(len(dtes)):
			indx = (12*(yer[i]-syr)+mon[i])-1
			gdates[indx] = mon[i]
			gpr[indx] = prec[i]
			print ('%3d %5d %3d %4d %3d %8.3f' %(i,yer[i],mon[i],indx,gdates[indx],gpr[indx]))

			# pause()
		if(len(dtes) < 144): print gdates
		pause()

;***************-999 to NaN
tave[tave == -999.99] = np.nan
;****************Nan to zeros
ndd = np.isnan(data)
data[ndd] = 0.0

;*******standard deviation
smn = np.nonzero((mon >=6) & (mon <=9))
sstd = np.nanstd(gprr[smn])

;*****************remove Nan from the data array
hprg = hprr[~np.isnan(hprr)]

;**********picking the seasonal months (smap_seasons_1x1_ncdf.py)
md = np.nonzero((yrmons >= 6) & (yrmons <= 9))
(or)
ind = np.where((d_time >= 1980) & (d_time <= 2014))		#waves_extract.py
print (d_time[ind])  

;****picking the Nan values
tdd = np.argwhere(np.isnan(data))

;******picking the non NaN indices values
tdd = np.where(~np.isnan(hprr))

;*****counting num of 'nan' values (era_12m_6m_multiregrsn.py)
ncnt = np.count_nonzero(np.isnan(data))

;*****counting num of 'non-nan' values(era_12m_6m_multiregrsn.py)
ncnt = np.count_nonzero(~np.isnan(data))

# ;********append two datasets with different shapes
A.shape = (600,200)
B.Shape = (600,300)
c = np.concatenate((A,B),axis=1)
c.shape = (600,500)

# ****stack two arrays into one (regression_multiple_mls.py) 
print(wdat.shape) = (128,)
print(sdat.shape) = (128,)
dup = np.stack((wdat,sdat),axis=1)
print(dup.shape) = (128,2)

dup = np.stack((wdat,sdat),axis=0)
print(dup.shape) = (2,128)

;****changing two dimnetional array to single array (qbo_multiregrsn.py)
# ***reading QBO winds at Singapore, 30 hpa level
qfle=dirpath+'qbo_singwinds_30hpa_1948-2017.dat'
print (os.path.join(dirpath, qfle))
dinput1 = np.loadtxt(os.path.join(dirpath,qfle),skiprows=1)
print(dinput1.shape)

qdate = dinput1[:,0]
qbodata = dinput1[:,1:13]
print(qbodata.shape)
data = (70,12)
newdata = data.flatten()
newdata = (840)

# ***3D to 1d array(cru_srftem_yrly_global_landreg.py)
sdata1 = np.squeeze(stemm[sy,:,:])
sdata = sdata1.ravel() (or)
sdat = sdata1.flatten()

(or) (cmip6_calcld_mnly_corr_rmse.py)
mdata = np.squeeze(np.concatenate(mdl6[mn,slat:elat,slon:elon]))

#mean of all models: Ensemble mean (contour_4x1.py)
ensmwin = np.mean(mdltrnd,axis=0)

;********picking the data indices same as GRACE time indices (test_tseries_decomposition.py)
;***comparing two arrays or same date values picking
last_month = np.min(np.array([d_time_dmi[-1], d_time[-1]]))
index = np.where((d_time >= tws_d_time[0]) & (d_time <= last_month))
d_time = d_time[index]

# ***another way two datasets same indices (cross_corr_wnd_tpht.py)
# ***here digital date to year and months creation and 
# picking the same indices from the dataset.
dindx = (np.empty(132,dtype=int))*np.nan
syr = np.fix(qdate[0])
for i in range(len(qdate)):
	monn = int(np.round((qdate[i]-np.fix(qdate[i]))*12.)+1)
	indx = int(12*(np.fix(qdate[i])-syr)+monn)-1
	dindx[i] = indx
	# print(i,monn,indx,mon[indx]); pause()
# print(mon); pause()
gindx = dindx[~np.isnan(dindx)]
gindx = gindx.astype(int)
print(len(gindx))
print(gindx); pause()

;***********Interpolation (test_tseries_decomposition.py)
tws_mass_interp = np.zeros(len(nino34))
for tt in range(len(nino34)):
	tws_mass_interp[tt] = np.interp(d_time[tt], tws_d_time, tws_mass)

;*************Plot the data and the fitted curve.
plt.plot(x, y, 'o', label='data')
xx = np.linspace(0, 9, 101)
yy = p[0] + p[1]*xx**2
plt.plot(xx, yy, label='least squares fit, $y = a + bx^2$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(framealpha=1, shadow=True)
plt.grid(alpha=0.25)
plt.show()

# **x or ytick number of tick (igra_blh_bar_con.py)
plt.locator_params(axis='y',nbins=5)
# ***fit + equation (igra_mnly_rchrdsn_blhplt_smth_fit.py)
# ****preparing fit eqn. (y = mx+c)
fit_eqn = str(np.poly1d(p,variable='x'))
# print(fit_eqn)
# print(p[0]);pause()

;*****output writing into a file (tai_tem_normalize_plt.py)
wfid =open('/Analysis/Tai_pr_tem/Results/tai_tem_normalized_1960-2012.txt','w')
wfid.write(' dyear   normtem '+ prgname+'\n')
for jj in range(mns):
	wfid.write('%8.3f %8.3f' %(months[jj],tavg[jj]))
	print('%8.3f %8.3f \n' %(months[jj],tavg[jj]))
wfid.close


****Making gap data to in order
****dates modifying index (read_noaa_index.py)
gdates = (np.empty(144, dtype=int))*np.NaN
gpr = np.empty(144)*np.NaN
# *****month date index creation
syr = 2002
for i in range(len(dtes)):
	indx = (12*(yer[i]-syr)+mon[i])-1
	gdates[indx] = mon[i]
	gpr[indx] = prec[i]
	print ('%3d %5d %3d %4d %3d %8.3f' %(i,yer[i],mon[i],indx,gdates[indx],gpr[indx]))

	# pause()
if(len(dtes) < 144): print gdates

pause()

List of list (bboxes_image_multiicons.py)
bboxes = []
cnames = []
for lines in lines_list:
    # print(lines.split())
    class_name, left, top, right, bottom = lines.split()
    bbox1 = [left,top,right,bottom]
    bbox = [int(i) for i in bbox1]
    print(class_name)
    print(bbox)
    bboxes.append(bbox)
    cnames.append(class_name)
    

;***invert axis or (reverse axis)for line plots or contour plots
;gfdl_am3_am4_lineplot.py
;gfdl_am4_zonmean_con.py
plt.gca().invert_yaxis()

;***attributes (cmip6_srftem_sesns_ncdf.py)
model = nc.source_id		#Attribute
	print (model)

******Multi ncdf file reading one way
#***input netcdf file location (multi_ncdf_read1.py)
dirpath = '/Volumes/UCI_ESS/CMIP6/stem_mnly/'
# fname ='ts_Amon_historical_GISS-E2-1-G_'    #don't put *.nc
fname ='ts_Amon_CNRM-CM6-1'

ncfiles = [os.path.join(dirpath, x) for x in os.listdir(dirpath) 
	if not x.startswith('.') and x.startswith(fname) and x.endswith('.nc')]

nfls = len(ncfiles)
for nf in range(0,nfls):
	# print os.path.join(dirpath, ncfiles[nf])
	print ncfiles[nf]

******Multi ncdf file reading one way
#***input netcdf file location (cmip6_srftem_sesns_ncdf.py)
dirpath = '/Volumes/UCI_ESS/CMIP6/stem_mnly/'
# fname ='ts_Amon_historical_GISS-E2-1-G_*.nc'
fname ='ts_Amon_CNRM-CM6-1_*.nc'
ncfiles = (fnmatch.filter(os.listdir(dirpath), fname ))
nfls = len(ncfiles)
for nf in range(0,nfls):
	print os.path.join(dirpath, ncfiles[nf])
	# print load_files[nf]

#**Attrbutes reading
model = nc.source_id (cmip6_srftem_sesns_ncdf.py)

# ***output file creation (cmip6_srftem_sesns_ncdf.py)
syrr = int('%d%s' %(hyer[0],'01'))
eyrr = int('%d%d' %(hyer[-1],12))
ofle = '/Analysis/CMIP6_mdls/Results/'+ model +"_srftem_seasons_"+ str(syrr)+"-"+str(eyrr)+".nc"
print (ofle)
pause()

# ***fill_value/missing value add into the netcdf file (write_ncdf_2var3de.py)
var_adat1 = rootgrp.createVariable('Temperature', 'f8', ('time','prlvls','lon'),fill_value=-32767)

# ***reading the missing valus (cmip5_srftem_yrly_regrid_ncdf.py)
missval = nc.variables['ts'].missing_value

#***changing the dimension of the data (91,144) to (144,91) gfdl_ncread_con.py
z150 = np.transpose(scli150)
# print(z150.shape)

# ***reverse oreder of an array
x = [10,20,30,40,50]
y = x[::-1]
or
x.reverse()

#***figure sides adjustment
# fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9, hspace=0.2)
# fig.tight_layout()
plt.subplots_adjust(hspace=0.175)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

#***colorbar (cmip6_srftem_sesns_ncdf_con.py)
# cbar_ax = fig.add_axes([0.71, 0.15, 0.02, 0.75])
cb = fig.colorbar(im, ax=ax[i],shrink=0.9,pad=0.01)

# cb.set_label('label', labelpad=-40, y=1.05, rotation=0)
cb.set_label("Surface Temp. (K)",labelpad=0.3,size=8)

# ***colorbar at specified location (ind_maxT_minT_meanT_selmonth_mesh_normal_con.py)
position = fig.add_axes([0.25,0.03,0.5,0.02])	#(left,bottom,width,height)
cb = fig.colorbar(im,cax=position,orientation='horizontal',drawedges=False)
cb.set_label('Temp. (C)',fontsize=5)

# *********colorbar vertical (contour_3x2.py)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.815, 0.15, 0.015, 0.7])
cb =fig.colorbar(im,cax=cbar_ax)
cb.set_label("Temp. (C)",labelpad=-1,size=12)

*****colorbar ticks manage (cmip56_clt_sesns_diff_8x3_con.py/igra_blh_bar_con.py)
if(i == 23):
	position = fig.add_axes([0.69,0.08,0.2,0.01])	#(left,bottom,width,height)
	cb = fig.colorbar(im,ax=ax[23],cax=position,orientation='horizontal',drawedges=False)
	cb.set_label('CMIP6-CMIP5 CLT (%)',labelpad=0.001,fontsize=5,fontweight='bold')
	cb.ax.tick_params(labelsize=5)
	# ***tick labels control
	from matplotlib import ticker
	tick_locator = ticker.MaxNLocator(nbins=6)
	cb.locator = tick_locator
	cb.update_ticks()

# ***string split and string length (cmip6_srftem_sesns_ncdf.py)
strlen = len((os.path.join(dirpath, ncfiles[0])))
syer = int((os.path.join(dirpath, ncfiles[0])).split('\t')[0][strlen-16:strlen-12])

# ****check this also (cmip6_srftem_sesns_ncdf_con.py)
strlen = len(ncfile)
model =ncfile.split('\t')[0][strlen-42:strlen-31]
print (strlen, model)

# ***Cubic spline interpolation (lon x lat)
cmip6_srftem_sesns_regrid_cubic.py
cmip6_srftem_sesns_regrid_ncdf.py

# ***bi-linear interpolation 
cmip6_srftem_sesns_bilinear_regrid_ncdf.py

# ***Interpolation for single array (imd_12m_6m_multiregrsn.py or cubic_test.py)
md, = np.where(data <= 0.0)
if(len(md) > 3):
    data[data < 0.0] = np.nan
    xx = np.arange(0,len(t_time))
    f = interp1d(xx, data, kind='cubic') 
    data1 = f(xx)
    plt.figure()
    plt.plot(t_time,data,'o', t_time,data1,'--')
    plt.show()

# ****Interpolation for GPS data sets (gps_wetprf_read.py)
# ***Interpolation
htmin = ht[0]
htmax = 12000.
hres =20.
m =(htmax-htmin)/hres
hn = htmin+(np.arange(0,m,dtype=float)+1)*hres

new_ht = np.linspace(htmin,htmax,m)
new_tem = sp.interpolate.interp1d(ht,tem,kind='cubic')(new_ht)


****count non zero only (cmip6_srftem_sesns_ncdf.py)
w1 = np.where((wdata >=100.) & (wdata < 500.))
wc = np.count_nonzero(~np.isnan(wdata[w1]))
if wc > 0: smwin[i,j] = np.nanmean(wdata[w1])
http://cmdlinetips.com/2018/04/an-introduction-to-altair-a-python-visualization-library/

# ***print years
years = int(ddays[0])+np.arange(0,nyrs)

;***cmip6_sesnavg_ncdf.py
years = syer + np.arange(0,(mns/12))
nyrs = len(years)
print(years,nyrs)
# ********************Making digital time serires using year and month
for t in range(len(year_dmi)):
    d_time_dmi[t] = digit_date(year_dmi[t], month_dmi[t])

;***IDL reform type
tem = np.flipud(np.squeeze(tem1[0,:,:])) (wind_vector_tem1.py or era_uv_lonlat_vector.py)

# plot wind vectors on projection grid. (wind_vector_tem1.py/era_uv_lonlat_vector.py)
# first, shift grid so it goes from -180 to 180 (instead of 0 to 360
# in longitude).  Otherwise, interpolation is messed up.
ugrid, newlons = shiftgrid(180., z150, lon, start=False)
vgrid, newlons = shiftgrid(180., m150, lon, start=False)
;***for wind vectors latitude should be 90S-90N then only it will work (wind_vector_tem1.py/era_uv_lonlat_vector.py)
;***change lat(90 to -90) to (-90 to 90) and changes dataset also (wind_vector_tem1.py)
lat = np.flipud(np.squeeze(nc1.variables['latitude'][:]))
tem1 = np.squeeze(nc1.variables['t'][:])
tem = np.flipud(np.squeeze(tem1[0,:,:]))

;****files order(cmip6_yrlyavg_ncdf.py)
ncfiles1 = (fnmatch.filter(os.listdir(dirpath), fname ))
ncfiles = sorted(ncfiles1)

;***line plotting multiple plots
;***plotting different ways (cmip6_prw_proj1.py)
# ******************Yearly projections
axs = {}
fig, ((axs[0],axs[1]),(axs[2],axs[3])) = plt.subplots(num=1, nrows=2, ncols=2, figsize=(6,8),dpi=150)

# fig, axs = plt.subplots(nrows=3, ncols=1,figsize=(6,8))
fig.subplots_adjust(hspace=0.1)

# *********************another way (cmip6_prw_proj.py)
# ******************Yearly projections
fig = plt.figure(figsize=(6,8))

# ***row1 cloumn 1 plot
ax1 = fig.add_subplot(321)		#(3,2,1)
ax1.plot(years, yrmneq_anm, color='blue', label='(20S-20N)')
ax1.axhline(0, color='black', lw=1)
ax1.set_ylim(np.min(yrmneq_anm), np.max(yrmneq_anm))
ax1.axes.xaxis.set_ticklabels([])
# ax.set_ylabel('cl',fontsize=12)
ax1.legend(loc='upper left', fontsize='small', ncol=2)

# ***row1 cloumn 2 plot
ax2 = fig.add_subplot(322)		#(3,2,2)
ax2.plot(years, yrmnnm_anm, color='brown', label='(20N-50N)')
ax2.axhline(0, color='black', lw=1)
ax2.set_ylim(np.min(yrmnnm_anm), np.max(yrmnnm_anm))
ax2.set_ylim(np.nanmin(yrmnnm_anm), np.nanmax(yrmnnm_anm))
ax2.axes.xaxis.set_ticklabels([])
ax2.legend(loc='upper left', fontsize='small', ncol=2)

(or) different way(cmip6_tcf_diffrntlat_allmdlsproj.py)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5,1.45),fontsize='small', ncol=3)
# # ***plot test
# import matplotlib.pyplot as plt
# plt.figure()
# # plt.plot(years,mneq,'r')
# plt.plot(years,yrmnnm_anm,'b')
# plt.show()

# ***Polynomial fit (lomb_yearly_py.py)
fit1 = np.polyfit(years,mneq,2)
fit = np.poly1d(fit1)
# ***plot test
k = np.linspace(1850.,2015.,500)
plt.plot(years,mneq,k,fit(k),'-')
plt.show()

# ***xticks (lomb_mnly_py.py)
plt.xticks(np.arange(0,156,12))

xticks (cmip56_diffrntlat_clim.py)
ax2.set_xticks(np.arange(1,13,2))

# ****This is one time declaration (bgi_slope_clim_violinplttmul.py)
def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels,fontsize=8)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Month')
    ax.yaxis.set_tick_params(labelsize=8)
# ***xticks fontsize (bgi_col_noaa_trmm_stat_bubleplt.py)
ax[i].tick_params(axis='y',which='major',labelsize=14)
# ;***picking the locations (station ids IGRA/global radiosonde datasets)
stn_id = lnes[ln].split(',')[0][6:11]   #(igra_ind_locs.py)
sel = np.where(ifnme == int(stn_id))[0]
if(len(sel) > 0):
	print(stn_id,lat,lon,elev,syer,eyer)
	print(len(sel),ifnme[sel,],stn_id)
	pause()

# ****separating all months to (years,12 (cmip6_pcaeofs_syrs.py)
id1 = 0
nms = 12
startY = syears[0]
ny = (syears[-1] - syears[0])+1
prw_all=np.empty((ny,nms,nlt,nln))
for i in range(startY,startY+ny):
	id2 = id1 + nms
	prw_all[i-startY,0:id2-id1,:,:] = prw_detrend[id1:id2,:,:]
	id1 = id2

#printing formats
print("Initial attributes: fps = {}, pos_msec ={}, pos_frames={}".format(fps, pos_msec, pos_frames))
print("{0}:{1}:{2}".format(con_hour,con_min,con_sec))
print("%d:%d:%d" %(con_hour,con_min,con_sec))
f'{minutes:0>2.0f}:{seconds:.3f}'

def convertMillis(millis):
    seconds=(millis/1000)%60
    minutes=(millis/(1000*60))%60
    hours=(millis/(1000*60*60))%24
    return seconds, minutes, hours

millis = 3177.4333333333334
con_sec, con_min, con_hour = convertMillis(int(millis))
print("{0}:{1}:{2}".format(con_hour,con_min,con_sec))

print("%d:%d:%d" %(con_hour,con_min,con_sec))


duration = 3177.4333333333334
minutes, seconds = divmod(duration / 1000, 60)

f'{minutes:0>2.0f}:{seconds:.3f}'
# ***text writing on a plot and superscript (era5_tem_dayvsht_60N_dly_con.py)
plt.xticks(np.arange(days[0],days[-1]+1,30))
plt.text(110,82,"60$^o$N")
(or)
tc6mean=234.56
syrr = ('%s%f' %('multi',tc6mean))

ax.text(-50.,90,'%s%5.2f%s' %('multi (',tc6mean,')'),fontsize=12,color='green') (cmip56_zonallyavg_plt.py)

# **** if else in Python (igra_ind_abl_mean.py)
print("actual data years: ",year[0],year[-1])
	if((year[0] < 1979) and (year[-1] >2000)):
		syr = np.where(year >= 1980)[0]
		for i in range(len(syr)):
			ng = int(syr[0])
			# print(i,ng,syr[i],year[ng]); pause()
			syer = year[ng]
	elif(year[0] > 1980):
		syer = year[0]
		ng =0
	elif ((year[0] > 1901) and (year[-1] < 1979)):
		syear = year[0]
		ng =0
# (or) (cmip6_srftem_sesns_act_bilinear_5x2_conn.py)
if(i == 8 or i == 9): m.drawmeridians(np.arange(60,360,60.),labels=[0,0,0,1],linewidth=0.2,fontsize=8)
	else: m.drawmeridians(np.arange(60,360,60.),labels=[0,0,0,0],linewidth=0.2,fontsize=8)

# ***if else statement (cmip5_srftem_yrly_regrid_ncdf.py)
if(len(wc) > 0): yravg[ny,i,j] = np.nanmean(sdata1[wc]) 
else: yravg[ny,i,j] = np.nan

# ***jday used in (igra_ind_clim_echyr.py)
# jday=[31,59,90,120,151,181,212,243,273,304,334,365]
# jday=[31,60,91,121,152,182,213,244,274,304,335,366]  #leap year

# ***different color text (cmip6_stem_proj.py)
ax1.text(1900.,7.5,r'1.5$^{o}$', ha="center", va="bottom", size="medium",color="red")
ax1.text(1925.,7.5,r'2.0$^{o}$', ha="center", va="bottom", size="medium",color="green")

# **** array (cmip6_pcaeofs_syrs.py)
clf = np.zeros((nfls,nlt,nln)) *np.nan

# ***Reading the IMD min,max and mean mean temp netcdf file (ind_max_min_dly2mnly_ncdf.py)
ncfile = '/Analysis/Temp_max_min/IMD_ind_maxt_mint_meant_cubic_1x1grid_mnly_195101-201512py.nc'
# Loading file
nc = Dataset(ncfile)
lon = nc.variables['lon'][:]

# ***modulus
if(np.fmod(i,3) == 0.):
	print(i)

# ***read variables and attributes in python (cmip6_surftem_annual_globally.py)
# ***reading the variables from ncdf file 
lon = nc.variables['lon'][:]		#longitude
lat = nc.variables['lat'][:]		#Latitude
tme = nc.variables['time'][:]		#months
clt = nc.variables['ts'][:]			#surface temperature

# ***reading the variable attributes (cmip6_surftem_annual_globally.py)
mval = nc.variables['ts'].missing_value

for attr in nc.ncattrs(): 
	print (attr, '=', getattr(nc, attr))

# *********single text file reading (cmip6_diffrntlat_htprfiles_plt.py)
clddata = np.loadtxt('/Volumes/UCI_ESS/CMIP_clouds/Results/calcld_cl_difrntlats_mnly_2006_2011.dat',skiprows=2)
print(clddata.shape)
data = clddata[:,0]

# *******multiple .txt or .dat files reading (cmip56_zonallyavg_1degfind_plt.py)
dirpath = '/Volumes/UCI_ESS/CMIP_clouds/Results/'
fname ='*_cmip6_srftem_echyr_65S-65N_1861-2005.txt'
files = (fnmatch.filter(os.listdir(dirpath), fname ))

nfls = len(files)
print(len(files))

for nf in range(0,nfls):
	print (os.path.join(dirpath, files[nf]))
	dinput2 = np.loadtxt(os.path.join(dirpath,files[nf]),skiprows=1)

#***changing grids (0-360) to (-180 to 180) (smap_mnly_indreg.py)
gdd = np.where(mlon > 180.)
mlon[gdd] = mlon[gdd]-360.
#mlon2 = (( mlon + 180 ) % 360 ) - 180

tmp=np.zeros(360)
tmp[:180] = mlon[180:]
tmp[180:] = mlon[:180]
mlon = tmp

tpp = np.zeros((180,360))
tpp[:,:180] = mask[:,180:]
tpp[:,180:] = mask[:,:180]
mask = tpp

# ***(cmip6_srftem_mnly_clim.py)
for jj in range(0,12):
	adata = np.squeeze(tmavgi[jj,:,:])
	adata = adata * mask 
	md = np.where(mask == 0.0)
	if(len(md) > 0): adata[md] = np.nan 
	avv[jj] = np.nanmean(adata)
	std[jj] = np.nanstd(adata)

# *************changing grids (-180 to 180) => (0 to 360) (cldcal_clt_regrid_ncdf.py)
# # ***changing (-180 to 180) => (0 to 360)
gdd = np.where(lon < 0.0)
lon[gdd] = lon[gdd]+360

tmp = np.zeros(180)
tmp[:90] = lon[90:]
tmp[90:] = lon[:90]
lon = tmp

# *****datasets changing (-180 to 180) => (0 to 360)
tpp = np.zeros((180,90))
tpp[:90,:] = savg[90:,:]
tpp[90:,:] = savg[:90,:]
savg = tpp

#data file reading (cmip6_htprfle_plt.py) or (cmip6_diffrntlat_htprfiles_plt.py)
# data structure (BCC-CSM2-MR_cl_mnly_2000-2014.txt)
 dyear     plev      mneq     mnnm     mnnh     mnsm     mnsh cmip6_slatavg_mnly.py
2000.000 1004.220    1.978    8.638   26.578   11.186   24.446 
2000.000  989.009    2.699    9.298   29.183   11.847   24.708 
2000.000  973.755    4.661   11.009   32.439   12.723   28.762 
2000.000  958.501    7.384   12.688   33.279   14.001   31.182 
2000.000  943.247    7.667   13.804   32.966   13.680   31.945 
for p in range(0,40):
	p1 = np.zeros(180,dtype=int)
	for m in range(0,180):
		p1[m] = p+(m*40)

# ****plot standard deviation (horizontal lines) and filling
ax1.plot(mneqa, ht, color='blue', label='($20^o$S-$20^o$N)')
ax1.errorbar(mneqa, ht, xerr=mneqs, capsize=3,fmt='o',markersize=3,elinewidth=0.7,markeredgewidth=0.5)

# filling: (cmip56_zonallyavg_plt2.py)
ax.fill_between(lat5,c5avg-c5std,c5avg+c5std,color='lightblue',alpha=0.45)

# filling: (cmip6_diffrntlat_htprfiles_plt.py)
ax1 = fig.add_subplot(231)		#(2,3,1)
ax1.plot(mneqa, ht, color='blue', label='($20^o$S-$20^o$N)')
# ax1.errorbar(mneqa, ht, xerr=mneqs, capsize=3,fmt='o',markersize=3,elinewidth=0.7,markeredgewidth=0.5)
ax1.fill_betweenx(ht,mneqa-2.*mneqs,mneqa+2.*mneqs,color='lightblue',alpha=0.45)

# ******y=mx+c slope and y-intercept (slope_intercept.py)
# main is usage of function
np.random.seed(12345678)
x = np.random.random(10)
y = 1.6*x + np.random.random(10)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print("slope: %f    intercept: %f" % (slope, intercept))

print("R-squared: %f" % r_value**2)

# ***otherway
r = stats.linregress(x,y)
print(r)
# the output is 
LinregressResult(slope=1.9448642607472155, intercept=0.26857823524544855, rvalue=0.8576118231957234, pvalue=0.00150893132301119, stderr=0.41235189090279994)
# if you want particular
print(r.slope)

(picking particular year value) (igra_corr_abl_sur.py)
for i in range(len(syer)):
    #     #surface data
        sy1, = np.where(suryer == float(syer[i]))
        ey1, = np.where(suryer == float(lyer[i]))
        print(sy1[11],ey1,'  ',ey1[11])

# ***climatological monthly mean (cmip6_srftem_mnly_clim.py)
print(hmon)
print(len(hmon))
tmavg = np.zeros((12,nlt,nln))*np.nan
for i in range(0,12):
	mn = i+1
	smn, = np.where(mn == smon)
	# print(smn); print(hmon[smn]);pause()
	for j in range(len(lat)):
		for k in range(len(lon)):
			rdata = np.squeeze(stem[smn,j,k])
			ravg = np.nanmean(rdata)
			tmavg[i,j,k] = ravg
print(tmavg.shape)

***random color selection(cmip6_imd_srftem_mnly_clim_plt.py)
number_of_colors = nfls
colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
# print(colors)

# ***digital date (cru_srftem_sesns_bilinear_syrs_regrid_ncdf.py)
hyer=np.empty(len(tme), dtype=int)
hmonths = np.empty((mns),dtype=float)
hmon=np.empty(mns, dtype=int)
for mn in range(mns):
	cmn = stmon+mn 
	hmon[mn] = ((cmn-1) % 12)+1
	if(mn > 0) and ((cmn-1) % 12 == 0):	yrr=yrr+1 
	hyer[mn] = yrr
	hmonths[mn] = yrr + ((cmn-1) %12)/12.
	# print(mn,hyer[mn],hmon[mn],hmonths[mn])

years = list(dict.fromkeys(hyer))
nyrs = len(years)

# create digital month (qbo_multiregrsn.py)
# for clarification (cmip_srftem_club_2x2_ncdf.pro or cmip_maxtem_split_1x1_ncdf.pro)
qmns = ((qdate[-1] - qdate[0])+1)*12
qddte = qdate[0] + ((np.arange(0,qmns))/12.)
print(qdate[0],qdate[-1],qmns)
print(qddte)

# ***digital day number
ddte = year + (np.arange(1,ds+1))/999.9
# ddte = np.around(year + (np.arange(1,ds+1,dtype=float))*1/1000., decimals =3)
# print(ddte); pause()

# ***digital data to year and month (cross_corr_wnd_tpht.py)
mon = (np.empty(132,dtype=int))*np.nan
syr = np.fix(qdate[0])
for i in range(len(qdate)):
	monn = int(np.round((qdate[i]-np.fix(qdate[i]))*12.)+1)

# ***array size (cmip56_stem_projs.py)
stem560 = (16,239)
print(len(stem560)) 
>16
print(len(stem560[0]))
>239

# ***contour purpose check these programs
era5_tem_dayvsht_60N_dly_con1.py
cmip56_clt_sesns_diff_8x3_con.py

# ***netcdf file writing two ways:
cmip5_srftem_yrly_regrid_ncdf.py
cmip6_hist_ssp585_samemdls_global_yrly_ncdf.py (string of array also write  into netcdf file)


# Gaussian fit
https://kippvs.com/2018/06/non-linear-fitting-with-python/
http://python4esac.github.io/fitting/examples1d.html


# ; ***reducing netcdf file to nearly 65% (isimip_daily_tas_ffdi_ncdf.pro)
;****Spaces are more important in SPAWN command
SPAWN, 'nccopy -d9 -s ' +ncfle1+'.nc'+' ' +ncfle1+'_rsize.nc'

; ;***delete the ps file 
spawn, 'rm '+ncfile1+'.nc' 


# ****mask data file reading (dat file) (hma_clim_har_imd_boxplt.py)
# ***BGI IMD covered region mask file reading
ascfle = open('/Analysis/TWS/ind_rivers/bgi_imd_mask_720x360.dat')
dinput = np.loadtxt(ascfle,skiprows=1)
mask = np.zeros((360,720),dtype=np.uint8)
dlon = 0.5
dlat = 0.5
for i,m in enumerate(dinput[:,2]):
	ilon = int(dinput[i,0]/dlon)-1
	ilat = int((90-dinput[i,1])/dlat)
	mask[ilat,ilon] = m

# ***checking the Lon and latitudes are in the region or polygon (bgi_col_noaa_trmm_stat_table.py)
# ***picking the lon and latitude points within the region or polygon
lons_lats_vect = np.column_stack((lon1, lat1)) # Reshape coordinates
polygon = Polygon(lons_lats_vect) # create polygon

# ***picking the datasets within the basin
npt = len(cor)
kk = -1
for i in range(len(cor)):
	slonn = stnlon[i]
	slatt = stnlat[i]
	
	point = Point(slonn,slatt) # create point
	# ***you can try three ways (contains or within or touch the polygon)
	# print(polygon.contains(point),point.within(polygon),polygon.touches(point))
	res = polygon.contains(point)
	if(res == True): 
		print(i,slonn,np.min(lon1),np.max(lon1),'   ',slatt,np.min(lat1),np.max(lat1))
		


# ***Linear regression techniques
https://intellipaat.com/blog/what-is-linear-regression/
https://datascienceplus.com/linear-regression-with-python/
https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155


Spyder 
Highlight portion and press F9 for run  

print("Number of images: {}".format(tflite_model_predictions.shape[0]))
print(f"Accuracy: \033[1m{format(acc*100)}%")

#***resize the image (bullet_crop_resize_originalimage.py (ipynb)
 ress1 = cv2.resize(imgcrop2, dsize=(24,14), interpolation = cv2.INTER_CUBIC)

***print all elements in the array
nums = np.zeros(100, dtype=int)
print(nums) or display(nums)
all numbers: print(*nums)

#create a dataframe (label_count.ipynb)
# df = pd.DataFrame(nums,lblcnt, columns=['labels','labelcount'])
df = pd.DataFrame({'labels':nums, 'labelcount':lblcnt})
df

#**** (grndtruth_txtfle2json.py)
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

****
How to remove square brackets for each item from a list [duplicate]
data = [[20.],[25.],[32.]]
newarr = [i[0] for i in data]
print(newarr)
*****
plot 25 images randomly selected images (show_sample function) in MNIST model

******minimum index value and convert into list
if(len(min_indices) <=0): 
            min_indic = np.argmin(xvalset[0:12])
            min_indices = [int(min_indic) for x in str(min_indic)]
            
#*****************
The program is in downloads and "./chkpoint/" mean is a directory in the downloads.
network.load_state_dict(torch.load('./chkpoint/July27_acc_0.9994_iter_265_model.pth'))

You can access .py program directly into jupyter notebook (week6_3_Decision_Tree.ipynb)
%run -i '/home/jayanthikishore/Desktop/Analysis/Work/ML_EIT/confusion_matrix_different_ways1.py'

#import .py file from different directory
import sys
sys.path.insert(0, '/local/mnt3/workspace3/Kishore/ML_project_dir/ML_project_env/test_folder/template_matching_trials/')
import imutils
###################How to import own function in python (IOU_map_functions.py)
from IOU_map_functions import file_lines_to_list

****string series to float array
data1 =[0.18431373 0.54509807 0.87843144 0.7843138]
print(type(data))
<class 'str'>
data = data[1:-1] # removing the paranthesis
floats_list = [float(item) for item in data.split()]

*******************************************************************(prediction_txtflejson.py)
tvmonitor 0.471781 0 13 174 244
cup 0.414941 274 226 301 265

with open(txtfle, "r") as fle:
    for line in fle:
        classname, acc, xmin, ymin, xmax, ymax = line.split() 
        
#read two columns ata time from 2D array
for i in range(2):
    data = img[:,[i,i+1]]
    data1 = np.concatenate(data)
**********************************************************************(json_explosion_cnt.py)
path = "C:/Users/kpangalu/Downloads/Audiodata_collection/Dec2022_audio/matfle2json/explosion-test-run1/"
#one way
# all_folders = glob.glob(os.path.join(path,'*.hdf'))
# print('Total folders: ',len(all_folders))

# for i in range(len(all_folders[0:2])):
#     print(all_folders[i])
#     #all_files = glob.glob(os.path.join(all_folders[i]))
#     all_files = os.listdir(all_folders[i])
#     for file in all_files:
#         print(file)

all_fldrs = os.listdir(os.path.join(path))
for echfldr in all_fldrs:
    all_files = os.listdir(os.path.join(path,echfldr))
    for file in all_files:
        fle1 = path+echfldr+'/'+file 
        if(file == 'annotation.json'):
            afile = open(fle1,'r')
            data = json.load(afile)
            print(data)
###############finding the json file and read the number of explosions in each file (check dirs and sub-dirs)
import os
import json

mpath = "C:/Users/kpangalu/Downloads/Audiodata_collection/Dec2022_audio/analysis/"
# mpath = "C:/Users/kpangalu/Downloads/Audiodata_collection/Dec2022_audio/matfle2json/explosion-test-run1/"
cnt = 0
for epath, subdirs, nfiles in os.walk(mpath):
    for file in nfiles:
        if file.endswith(".json"):
            selfile = os.path.join(epath,file)
            sfile = json.load(open(selfile,'r'))
            num_explsions= sfile["Num_ROIs"] 
            cnt += num_explsions
            
print('Total number of explosions: ',cnt)