import pandas as pd 

# Reading data from Tam 1981
def readData_Tam1981():
    d_WT = {} # WT animals
    d_WT['nSS'] = pd.read_csv('../experimental_data/Tam1981/Fig2_WT.csv' , index_col=None, header=0)
    d_WT['PSM'] = pd.read_csv('../experimental_data/Tam1981/Fig10_WT.csv', index_col=None, header=0)
    d_WT['seg'] = pd.read_csv('../experimental_data/Tam1981/Fig13_WT.csv', index_col=None, header=0)
    d_WT['L']   = pd.read_csv('../experimental_data/Tam1981/Fig14_WT.csv', index_col=None, header=0)
    d_WT['G']   = pd.read_csv('../experimental_data/Tam1981/Fig15_WT.csv', index_col=None, header=0)
    d_WT['f']   = pd.read_csv('../experimental_data/Tam1981/Fig16_WT.csv', index_col=None, header=0)

    dMMC={} # MMC-treated animals
    dMMC['nSS'] = pd.read_csv('../experimental_data/Tam1981/Fig2MMC.csv' , index_col=None, header=0)
    dMMC['PSM'] = pd.read_csv('../experimental_data/Tam1981/Fig10MMC.csv', index_col=None, header=0)
    dMMC['seg'] = pd.read_csv('../experimental_data/Tam1981/Fig13MMC.csv', index_col=None, header=0)
    dMMC['L']   = pd.read_csv('../experimental_data/Tam1981/Fig14MMC.csv', index_col=None, header=0)
    dMMC['G']   = pd.read_csv('../experimental_data/Tam1981/Fig15MMC.csv', index_col=None, header=0)
    dMMC['f']   = pd.read_csv('../experimental_data/Tam1981/Fig16MMC.csv', index_col=None, header=0)

    #  data in min or somite stage and um
    d_WT['nSS'].values[:,0] = d_WT['nSS'].values[:,0]*24*60
    d_WT['L'  ].values[:,0] = d_WT['L'  ].values[:,0]*24*60
    d_WT['L'  ].values[:,1] = d_WT['L'  ].values[:,1]*1e+3
    d_WT['G'  ].values[:,0] = d_WT['G'  ].values[:,0]*24*60.0
    d_WT['G'  ].values[:,1] = d_WT['G'  ].values[:,1]*(1e+3)/(24*60.0)
    d_WT['PSM'].values[:,1] = d_WT['PSM'].values[:,1]*1e+3
    
    dMMC['nSS'].values[:,0] = dMMC['nSS'].values[:,0]*24*60
    dMMC['L'  ].values[:,0] = dMMC['L'  ].values[:,0]*24*60
    dMMC['L'  ].values[:,1] = dMMC['L'  ].values[:,1]*1e+3
    dMMC['G'  ].values[:,0] = dMMC['G'  ].values[:,0]*24*60.0
    dMMC['G'  ].values[:,1] = dMMC['G'  ].values[:,1]*(1e+3)/(24*60.0)
    dMMC['PSM'].values[:,1] = dMMC['PSM'].values[:,1]*1e+3        
    
    return d_WT, dMMC


# Reading data from Lauschke et al 2013 
def readData_Lauschke2013():
    d37 = {}       # data from temperature = 37C
    d37['len'] = pd.read_csv('../experimental_data/Lauschke_data/Fig3a.csv', index_col=None, header=0)
    d37['vel'] = pd.read_csv('../experimental_data/Lauschke_data/Fig3b.csv', index_col=None, header=0)
    
    d33 = {}       # data from temperature = 37C
    d33['len'] = pd.read_csv('../experimental_data/Lauschke_data/Fig3e.csv', index_col=None, header=0)
    d33['vel'] = pd.read_csv('../experimental_data/Lauschke_data/Fig3f.csv', index_col=None, header=0)
    
    return d37, d33
    
    
# Reading data from Gomez et al 2008    
def readData_Gomez2008():
    d = {}
    d['snake']   = pd.read_csv('../experimental_data/Gomez2008/Snake_PSM_segment.csv',  index_col=None, header=0)
    d['chicken'] = pd.read_csv('../experimental_data/Gomez2008/Chicken_PSM_segment.csv',index_col=None, header=0)
    d['mouse']   = pd.read_csv('../experimental_data/Gomez2008/Mouse_PSM_segment.csv',  index_col=None, header=0)

    return d
