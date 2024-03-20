# DATASET CLEANING
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""
import os
from functions.def_functions import add_index_column,date_anio,day_week,sort_columns,rounding_data
from paths.paths import path_base,folder_df_clean,file_df_clean

def mod_dtset_clean(df_data,start_date,endin_date):
    print('\n')
    print(f'START MODUL mod_dtset_clean')
    
    #Restart dataframe jic
    restart_dataframes = True  
    if 'df_clean' in locals() and restart_dataframes:del df_clean  # delete dataframe if exits 
            
    df_clean = df_data.copy()
    df_clean = add_index_column(df_clean)
    df_clean = date_anio(df_clean)
    df_clean = day_week(df_clean)
    df_clean = sort_columns(df_clean)
    df_clean = rounding_data(df_clean)
        
    # SAVE FILE with start_date and endin_date suffixes
    if not os.path.exists(os.path.join(path_base, folder_df_clean)):os.makedirs(os.path.join(path_base, folder_df_clean))
    file_df_clean = f"df_clean_{start_date}_{endin_date}.csv"
    excel_file_path = os.path.join(path_base, folder_df_clean, file_df_clean)
    df_clean.to_csv(excel_file_path, index=False)
    
    print(f'ENDIN MODUL mod_dtset_clean\n')
    return df_clean

#if __name__ == "__main__":
    #Este bloque se ejecutará solo si el script se ejecuta directamente,
    #no cuando se importa como un módulo.
#    mod_dtset_clean(df_data,start_date,endin_date)
