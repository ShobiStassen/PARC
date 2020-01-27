'''
Latest version of Lung Cancer classifier using ALPHA and APT.
Can choose between Jan, May and June datasets
'''

import numpy as np
import time
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 350
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from pandas import ExcelWriter
import time
import sklearn.cluster

import os.path


print(os.path.abspath(sklearn.cluster.__file__))

def write_list_to_file(input_list, filename):
    """Write the list to file."""

    with open(filename, "w") as outfile:
        for entries in input_list:
            outfile.write(str(entries))
            outfile.write("\n")

def get_data( n_eachsubtype=None, randomseedval=1):

    # ALL FEATURES EXCLUDING FILE AND CELL ID:
    feat_cols = ['Area', 'Volume', 'Circularity', 'Attenuation density', 'Amplitude var', 'Amplitude skewness',
                 'Amplitude kurtosis', 'Dry mass', 'Dry mass density', 'Dry mass var', 'Dry mass skewness', 'Peak phase',
                 'Phase var', 'Phase skewness', 'Phase kurtosis', 'DMD contrast 1', 'DMD contrast 2', 'DMD contrast 3',
                 'DMD contrast 4', 'Mean phase arrangement', 'Phase arrangement var', 'Phase arrangement skewness',
                 'Phase orientation var', 'Phase orientation kurtosis', 'Focus factor 1', 'Focus factor 2']

    # January, May(H526) and June (HCC827)

    df_h2170 = make_subtype_df('h2170','h21702018Jan23_gatedH2170',0,n_eachsubtype, randomseedval, HTC_filename ="\\\\Desktop-u14r2et\\G\\2018Jan23\\" ) # same path 2170_Raw, 28 x 302,635
    df_h1975 = make_subtype_df('h1975','h19752018Jan23_gatedH1975',1,n_eachsubtype, randomseedval, HTC_filename= 'F:\\') #60447 *same path*
    df_h526 = make_subtype_df('h526','H5262018May24_gatedH526',2,n_eachsubtype, randomseedval, HTC_filename = 'E:\\2018May24_cancer\\')#375889 #WAS 562 until Jan16 \\Desktop-p9kngca\e
    df_h520 = make_subtype_df('h520', 'h5202018Jan03_gatedH520',3, n_eachsubtype, randomseedval,HTC_filename='\\\\DESKTOP-H5E5CH1\\DVDproc - 2 (G)\\20180103 H520\\')  # 451208 *same path*
    df_h358 = make_subtype_df('h358','h3582018Jan03_gatedH358',4,n_eachsubtype, randomseedval, HTC_filename='\\\\DESKTOP-H5E5CH1\\DVDproc - 2\\2018Jan3_h358_520_526\\')#170198 *same path*
    df_h69 = make_subtype_df('h69','h692018Jan23_gatedH69',5,n_eachsubtype, randomseedval,HTC_filename= 'F:\\') #130075 *same path*
    df_hcc827 = make_subtype_df('hcc827','hcc8272018Jun05_gatedHcc827',6,n_eachsubtype, randomseedval, HTC_filename='E:\\2018Jun05_lungcancer\\') #same path
    '''
    # May, January (H520), June (HCC827)
    df_h2170 = make_subtype_df('H21702018May24_gatedH2170',0,n_eachsubtype, randomseedval, HTC_filename ='\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\' ) #h2170_Raw, 28 x 302,635
    df_h1975 = make_subtype_df('H19752018May24_gatedH1975',1,n_eachsubtype, randomseedval, HTC_filename= '\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\') #60447
    df_h526 = make_subtype_df('H5262018May24_gatedH526',2,n_eachsubtype, randomseedval, HTC_filename = '\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\')#375889
    df_h520 = make_subtype_df('h5202018Jan03_gatedH520', 3, n_eachsubtype, randomseedval, HTC_filename='\\\\DESKTOP-H5E5CH1\\DVDproc - 2 (G)\\20180103 H520\\')  # 451208
    df_h358 = make_subtype_df('H3582018May24_gatedH358',4,n_eachsubtype, randomseedval, HTC_filename='\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\')#170198
    df_h69 = make_subtype_df('H692018May24_gatedH69',5,n_eachsubtype, randomseedval,HTC_filename= '\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\') #130075
    df_hcc827 = make_subtype_df('hcc8272018Jun05_gatedHcc827',6,n_eachsubtype, randomseedval, HTC_filename='\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\')
    '''
    # June, May(H526)
    '''
    df_h2170 = make_subtype_df('h21702018Jun05_gatedH2170',0,n_eachsubtype, randomseedval, HTC_filename ="\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\" ) #h2170_Raw, 28 x 302,635
    df_h1975 = make_subtype_df('h19752018Jun05_gatedH1975',1,n_eachsubtype, randomseedval, HTC_filename= '\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\') #60447
    df_h526 = make_subtype_df('H5262018May24_gatedH526',2,n_eachsubtype, randomseedval, HTC_filename = '\\\\Desktop-p9kngca\htc-4(b)\\2018May24_cancer\\')#375889
    df_h520 = make_subtype_df('h5202018Jun05_gatedH520',3,n_eachsubtype, randomseedval, HTC_filename= '\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\')#451208
    df_h358 = make_subtype_df('h3582018Jun05_gatedH358',4,n_eachsubtype, randomseedval, HTC_filename='\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\')#170198
    df_h69 = make_subtype_df('h692018Jun05_gatedH69',5,n_eachsubtype, randomseedval,HTC_filename= '\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\') #130075
    df_hcc827 = make_subtype_df('hcc8272018Jun05_gatedHcc827', 6, n_eachsubtype, randomseedval, HTC_filename='\\\\Desktop-p9kngca\\htc-4(b)\\2018Jun05_lungcancer\\')
    '''
    frames = [df_h2170, df_h1975, df_h526, df_h520 ,df_h358, df_h69,df_hcc827]
    #frames = [df_h2170,df_h1975,df_h526,df_h520,df_h358]
    df_all = pd.concat(frames, ignore_index=True,sort=False)


    # EXCLUDE FLUOR FEATURES

    df_all[feat_cols] = (df_all[feat_cols] - df_all[feat_cols].mean()) / df_all[feat_cols].std()

    X_txt = df_all[feat_cols].values
    print('size of data matrix:', X_txt.shape)

    label_txt = df_all['class'].values
    true_label = np.asarray(label_txt)
    true_label = true_label.astype(int)
    print('data matrix size', X_txt.size)
    print('true label shape:', true_label.shape)

    return df_all, true_label, X_txt, feat_cols



def make_subtype_df(str_subtypename, subtype_name, class_val,n_eachsubtype=None, randomseedval = 1, HTC_filename='dummy'):
    print('getting ', subtype_name)
    #print(randomseedval, ' is the randomseed value')
    subtype_raw = scipy.io.loadmat(
        '/home/shobi/Thesis/Data/ShobiGatedData/LungCancer_ShobiGatedData_cleanup/'+subtype_name+'.mat')
    subtype_struct = subtype_raw[subtype_name]
    df_subtype = pd.DataFrame(subtype_struct[0, 0]['cellparam'].transpose().real)
    subtype_features = subtype_struct[0, 0]['cellparam_label'][0].tolist()
    subtype_fileidx = pd.DataFrame(subtype_struct[0, 0]['gated_idx'].transpose())
    flist = []
    for element in subtype_features:
        flist.append(element[0])
    df_subtype.columns = flist
    subtype_fileidx.columns = ['filename', 'index']
    #print('shape of fileidx', subtype_fileidx.shape)
    df_subtype['cell_filename'] = HTC_filename + subtype_name.split('_')[0] + '_' + subtype_fileidx["filename"].map(
            int).map(str)
    if class_val == 3 or class_val==4:
        df_subtype['cell_filename'] = HTC_filename + subtype_name.split('_')[0] + '_' + subtype_fileidx["filename"].map(
            int).map(str)+'mat'
        #print('class val is ',class_val)

    df_subtype['cell_idx_inmatfile'] = subtype_fileidx["index"]#.map(int).map( str)  # should be same number as image number within that folder
    df_subtype['cell_tag'] = subtype_name.split('_')[0] + subtype_fileidx["filename"].map(int).map(str) + '_' + \
                              subtype_fileidx["index"].map(int).map(str)
    df_subtype['label'] = class_val
    df_subtype['class'] = class_val
    df_subtype['class_name'] = str_subtypename

    #print('shape before drop dups', df_subtype.shape)
    df_subtype = df_subtype.drop_duplicates(subset = flist, keep='first')
    df_subtype.replace([np.inf, -np.inf], np.nan).dropna()
    #print('shape after drop dups', df_subtype.shape)
    #print(df_subtype.head(5))
    #if class_val ==6:
        #df_subtype = df_subtype.sample(frac=1).reset_index(drop=True)
    if n_eachsubtype ==None:
        df_subtype = df_subtype.sample(frac=1).reset_index(drop=False)
    if n_eachsubtype !=None:
        if n_eachsubtype < df_subtype.shape[0]:
            df_subtype = df_subtype.sample(frac=1, random_state = randomseedval).reset_index(drop=False)[0:n_eachsubtype]
        else: df_subtype = df_subtype.sample(frac=1, random_state = randomseedval).reset_index(drop=False)
    print(df_subtype.shape)
    return df_subtype

def main():
    df_all_train, true_label_train, X_txt_train, feat_cols = get_data( n_eachsubtype=1000, randomseedval=3)
    df_all_test, true_label_test, X_txt_test, feat_cols = get_data( n_eachsubtype=200, randomseedval=4)
if __name__ == '__main__':
    main()