;+
; NAME:
;
;  make_online_emissions
;
; PURPOSE:
;
;  Create input for the new COSMO online emissions module in the form of a set
;  of netcdf files. For details, see document Online_emissions_for_COSMO_v1.2.docx.
;
;  Stored are:
;    a) gridded 2D emissions per emission category
;    b) temporal profiles per emission category (and partly per country)
;    c) vertical profiles per emission category  
;    d) country masks
;
; CATEGORY:
;
;  Emission processing, SMARTCARB, Satellite CO2, CO, NO2, online emissions
;
; CALLING SEQUENCE:
;
;   make_online_emissions,yyyy,configuration,datorigin=datorigin,$
;                          refine=refine,data=data,grid=grid,mask=mask
;
; INPUTS:
;
;   yyyy (STRING)  : YYYY: The year for which to generate files
;
; KEYWORD PARAMETERS:
;
;   configuration    : A structure with the following boolean switches:
;                       {CH4_EDGAR     :0B,$   ; EDGAR v4.2 CH4 emissions
;                        CH4_TNOMACC   :0B,$   ; TNO/MACC-3 CH4 emissions
;                        CO2_EDGAR     :0B,$   ; EDGAR v4.2 CO2 emissions
;                        CO2_TNOMACC   :1B,$   ; TNO/MACC-3 CO2 emissions
;                        CO_EDGAR      :0B,$   ; EDGAR v4.2 CO emissions
;                        CO_TNOMACC    :1B,$   ; TNO-MACC CO emissions
;                        NOx_EDGAR     :0B,$   ; EDGAR v4.2 NOx emissions
;                        NOx_TNOMACC   :1B,$   ; TNO-MACC NOx emissions
;                        Berlin        :0B,$   ; replace TNO-MACC over Berlin by Berlin invent.
;                        domain        :'',$   ; leave empty for SmartCarb
;                        gridname      :''}    ; the grid name
;
;                      Note: for each parameter (CH4, CO2, CO, NOx) one data source for the EU
;                      domain can be set to true (=1B).
;                      Default values are indicated above.
;
;                      Currently supported grid names are:
;                      'Berlin-big'
;                      Based on the grid name, the code generates a grid structure 
;                      grid = {nx:nx,ny:ny,dx:0.,dy:0.,xmin:0.,ymin:0.,pollon:0.,pollat:0.,$;
;                       xcoos:FltArr(nx),ycoos:FltArr(ny),name:''}
;
;   refine           : factor for refining TNO/MACC emissions to fit into high-resolution
;                      COSMO grid. Default for the 1x1 km2 COSMO grid is 16.
;
; OUTPUTS:
;
;   The following outputs can also be provided as inputs, in which case the reading
;   of emission files is skipped.
;
;   data             : structure containing emission fields for all species and SNAPs
;   grid             : structure describing the COSMO grid
;   mask             : The country mask with EMEP country IDs
;   datorigin       : string describing the origin of the emission data
;
; COMMON BLOCKS:
;
;  none
;
; SIDE EFFECTS:
;
;  generates three netcdf files with
;   a) gridded emissions
;   b) time profiles
;   c) vertical emission scaling profiles
;
; RESTRICTIONS:
;
;  relies on a number of routines of the CarboCount emission processor
;  ccEmissionPreprocessor/
;
; PROCEDURE:
;
;  none
;
; EXAMPLE:
;
;  ;; to create an annu the new online emission module
;  configuration = {CH4_EDGAR:0B,CH4_TNOMACC:0B,$
;                  CO2_EDGAR:0B,CO2_TNOMACC:1B,$
;                  CO_EDGAR :0B,CO_TNOMACC:1B,$
;                  NOx_EDGAR:0B,NOx_TNOMACC:1B,$
;                  Berlin        : 0B,$
;                  domain        :'',$  
;                  gridname      :'Berlin-coarse'} 
;  make_online_emissions,'2015',configuration,datorigin=datorigin,$
;                          refine=refine,data=data,grid=grid,mask=mask
;
;
; MODIFICATION HISTORY:
; 
;   (c) Dominik Brunner
;   Swiss Federal Laboratories for Materials Science and Technology
;   Empa Duebendorf, Switzerland
;
;   V1: DB, 10 Jan 2018
;       First implementation, adapted from make_smartcarb_emissions.pro
;   V2: DB, 14 Jul 2018
;       Strongly revised, first fully functional version. Nesting of Berlin emissions
;       remove to simplify routine.
;-

@cc_environment                 ; paths
@cc_tools                       ; toolbox mainly from Christoph Knote
@cc_timeprofiles                ; time profile routines
@cc_annual_scaling              ; scaling to different years
@map_latlongrid2cosmo.pro
@map_pointsource2cosmo.pro
PRO make_online_emissions,yyyy,configuration,datorigin=datorigin,$
                          refine=refine,data=data,grid=grid,mask=mask
  
  cc_environment
  
  COMMON cc_paths
  
  species = ['CO2']             ;,'CO','NOX']
  snap = [1,2,34,5,6,70,8,9,10]

  
  outputPath = '/project/brd134/smartcarb/online_emissions/'

  edgarversion  = 'v42_FT2010'  ; use this version of EDGAR data
  maccversion = 'III'           ; use this version for TNO/MACC data
  seconds_start = systime(/seconds)
  
  IF n_elements(refine) EQ 0 THEN refine=16

  ;;*******************************************************************
  ;; check for input
  ;;*******************************************************************
  IF n_elements(yyyy) EQ 0 THEN BEGIN
     message,'parameter yyyy missing',/continue
     RETURN
  ENDIF

  IF n_elements(configuration) EQ 0 THEN BEGIN
     print,'parameter configuration missing in call; use default setting for Berlin-big'
     configuration = {CH4_EDGAR:0B,CH4_TNOMACC:0B,$
                      CO2_EDGAR:0B,CO2_TNOMACC:1B,$
                      CO_EDGAR :0B,CO_TNOMACC:0B,$
                      NOx_EDGAR:0B,NOx_TNOMACC:0B,$
                      Berlin:0B,$
                      domain        :'',$  
                      gridname      :'Berlin-big'} 
  ENDIF

  tn = tag_names(configuration)
  index = WHERE(tn EQ 'GRIDNAME',cnt)
  IF cnt EQ 0 THEN BEGIN
     message,'tag "GRIDNAME" missing in structure configuration',/continue
     RETURN
  ENDIF
  
  ;;********************************************************
  ;; get grid for given configuration
  ;;********************************************************
  grid = cc_grid_configurations(configuration.gridname,ok=ok)
  IF NOT ok THEN return
  nocut = STRMID(configuration.gridname,0,5) EQ 'INGOS'

  IF grid.pollat NE 90. THEN projection = 'rotated' ELSE projection = 'regular'

  datorigin=' '

  IF STRLEN(yyyy) NE 4 THEN BEGIN
     print,'yyyy must be indicate a single year in format YYYY'
     RETURN
  ENDIF

  IF n_elements(data) NE 0 AND n_elements(mask) NE 0 THEN GOTO,skipread

  ncat = n_elements(snap)

  ;; loop over the species and SNAP categories and read data
  free,data
  FOR i=0,n_elements(species)-1 DO BEGIN

     FOR k=0,ncat-1 DO BEGIN
        free,euemis
   
        CASE snap[k] OF
           70: snapstr = '07'
           34: IF species[i] EQ 'CH4' THEN snapstr = '03' ELSE snapstr = '04'
           ELSE: snapstr = string(snap[k],format='(i2.2)')
        ENDCASE

        ;;*****************************************************************
        ;; process TNO/MACC emissions for EU domain
        ;;*****************************************************************
        CASE species[i] OF
           'CH4': dotnomacc = configuration.ch4_tnomacc
           'CO2': dotnomacc = configuration.co2_tnomacc
           'CO': dotnomacc = configuration.co_tnomacc
           'NOX': dotnomacc = configuration.nox_tnomacc
           ELSE: dotnomacc = 0
        ENDCASE
        
        IF dotnomacc THEN BEGIN
           IF k EQ 0 THEN datorigin = datorigin + species[i]+' EU:TNO/MACC-'+maccversion+'; '
           print,'Processing TNO/MACC-'+maccversion+' '+species[i]+' SNAP'+snapstr+$
                 ' emissions for EU domain'
           euemis = cc_get_tnomacc_emissions(species[i],yyyy+'0101',$
                                             snap=snap[k],version=maccversion,/constant,$
                                             /pointsources,/donotdel,ok=ok)
           IF ok EQ 0 THEN stop
        ENDIF

        ;;*****************************************************************
        ;; process EDGAR emissions for EU domain
        ;;*****************************************************************
        CASE species[i] OF
           'CH4': doedgar = configuration.ch4_edgar
           'CO2': doedgar = configuration.co2_edgar
           'CO': doedgar = configuration.co_edgar
           'NOX': doedgar = configuration.nox_edgar
           ELSE: doedgar = 0
        ENDCASE
        
        IF doedgar THEN BEGIN
           IF k EQ 0 THEN datorigin = datorigin+species[i]+' EU:EDGAR '+edgarversion+'; '
           print,'Processing EDGAR '+edgarversion+' '+species[i]+' SNAP'+snapstr+$
                 ' emissions for EU domain'
           euemis = cc_get_edgar_emissions(species[i],yyyy+'0101',version=edgarversion,$
                                           snap=snap[k],/constant,nocut=nocut,ok=ok)
           IF ok EQ 0 THEN stop
        ENDIF
        
        IF n_elements(euemis) EQ 0 THEN goto,nextcat
      
        ;;*****************************************************************
        ;; map EU domain onto cosmo grid
        ;;*****************************************************************
        print,'mapping EU '+species[i]+' SNAP'+snapstr+' emissions onto COSMO grid'
        map_latlongrid2cosmo,reform(euemis.emiss[0,0,0,*,*]),euemis.lons,euemis.lats,$
                             datout=outarea,lmgrid=grid,refine=refine
        tn = tag_names(euemis)
        haspointemis = abs(total(strpos(tn,'PTLONS'))) LT n_elements(tn)
        IF haspointemis THEN BEGIN
           ;; map point sources separately onto COSMO grid
           print,'mapping point sources'
           map_pointsources2cosmo,reform(euemis.ptemiss[0,0,0,*]),euemis.ptlons,euemis.ptlats,$
                                  datout=outpoint,lmgrid=grid
        ENDIF
      
        ;;*****************************************************************
        ;; create output data structure
        ;;*****************************************************************
        IF n_elements(data) EQ 0 THEN BEGIN 
           data = Create_Struct(species[i]+'_'+snapstr+'_area',outarea)
        ENDIF ELSE BEGIN
           data = Create_Struct(species[i]+'_'+snapstr+'_area',outarea,data)
        ENDELSE
        IF haspointemis THEN $
           data = Create_Struct(species[i]+'_'+snapstr+'_point',outpoint,data)

        nextcat:
     
     ENDFOR                     ; loop over categories


  ENDFOR                        ; loop over species

  ;; create country masks
  tmask = make_gridded_country_masks(grid,/hires,nfine=nfine)
  mask = IntArr(grid.nx,grid.ny)
  FOR i=0,n_elements(tmask)-1 DO BEGIN
     emepid=getemepcountryid(tmask[i].id)
     index=WHERE(tmask[i].mask GT 0,cnt)
     IF cnt GT 0 THEN mask[index]=emepid
  ENDFOR

skipread: 

  ;;*****************************************************
  ;;  dump to netCDF files
  ;;*****************************************************
  sc_dump_online_files, yyyy, data, grid, mask, configuration, datorigin, species

  seconds_end = systime(/seconds)
  print,'Emission processing for month ',yyyy,' took ',seconds_end-seconds_start,' seconds'
  
END
