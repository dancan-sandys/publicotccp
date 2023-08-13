import io
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import re
import numpy as np
from randomforest.models import UploadedFolder
from randomforest.train import train


comp = ["Area", "BoundingBoxAA", 'BoundingBoxOO', 'Distance from Origin Ref',
    'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Number of Triangles', 'Number of Voxels', 
    'Position Reference Frame', 'Shortest Distance to Surfac', 'Sphericity', 'Volume']  # Replace this with your desired sheet names

param_labels = ["Area", "BB AA X","BB AA Y","BB AA Z",'BB OO X',
    'BB OO Y', 'BB OO Z', 'Dist Origin', 'Oblate','Prolate',
    'Triangles', 'Voxels', 'Position X', 'Position Y', 'Position Z', 'Dist Surface',
'Sphericity', 'Volume', 'CI', 'BI AA', 'BI OO', 'Polarity'] #Note: this variable gets added to and reordered

def extract_cell_number(active_file, match):
    cellnum_filename = active_file.replace(match, '')
    cellnum_str = re.findall(r'cell\d*', cellnum_filename)
    
    if len(cellnum_str) > 0:
        cellnum = float(re.findall(r'\d+', cellnum_str[0])[0])
    else:
        cellnum = None
    
    return cellnum

# Function 1: parameter_import
def parameter_import():
    
    files = UploadedFolder.objects.all()
    
    X = np.empty((0, 0))  
    y = []
    X_labels = []
    cell_list = []
    celllines = []

    for i, file in enumerate(files):
        print(i)
        
        # Read the content of the UploadedFolder object into memory using io.BytesIO
        active_file_data = io.BytesIO(file.folder.read())

        # Work with the file data in memory 

        # Get the sheet names from the file data in memory
        full_sheets = pd.ExcelFile(active_file_data).sheet_names
       
        # Filter the sheets
        sheets_index = [idx for idx, name in enumerate(full_sheets) if name in comp]
        
        #Close the file in memory
        #     active_file_data.close


        #initialize X_temp
        X_temp = np.empty((0, 0))
        

        

        for k, var in enumerate(sheets_index):
            active_sheet = full_sheets[var]
       
            if "BoundingBox" in active_sheet:
                print("Bounding Box")
                imported_table = pd.read_excel(active_file_data, sheet_name=active_sheet, usecols="A:C")
                temp = imported_table.to_numpy()
                # X_temp = np.hstack([X_temp, temp])
                if X_temp.shape[0] == 0:
                    X_temp = temp

                else:
                    # Check if the number of rows in temp is the same as X_temp
                    if X_temp.shape[0] == temp.shape[0]:
                        X_temp = np.hstack([X_temp, temp])  # Append temp horizontally to X_temp                    
                        # print(X_temp)
                    else:
                        # Handle mismatch in the number of rows (e.g., raise an error or skip the temp array)
                        print("Mismatch in the number of rows. Skipping this temp array.")


            elif "Distance from Origin" in active_sheet or "Position" in active_sheet:
                
                if "Distance from Origin" in active_sheet:
                    print("Distance from origin 1")
                    imported_table = pd.read_excel(active_file_data, sheet_name=active_sheet, usecols="E:E")
                else:
                    print("Distance from origin 2")
                    imported_table = pd.read_excel(active_file_data, sheet_name=active_sheet, usecols="G:G")
                cellnum_data = imported_table.to_numpy().flatten()

                # Your logic for matching cell numbers and storing in cell_list goes here

                # cellnum_index = np.isin(cellnum_data, cellnum)
                
                # cellnum_index = np.isin(cellnum_data, 1)
                # if cellnum_index.sum() == 0:
                    # error = f"Cell Reference Undetected. Automatically importing all reference data: {cellnum_filename}"
                    # print(error)
                    # cellnum_index[:] = True

                if "Distance" in active_sheet:
                    print("Distance from origin 3")
                    imported_table = pd.read_excel(active_file_data, sheet_name=active_sheet, usecols="A:A")
                    temp = imported_table.to_numpy()
                    X_temp = np.hstack([X_temp, temp[5]])

                    if X_temp.shape[0] == 0:
                        X_temp = temp[5]

                    else:
                        # Check if the number of rows in temp is the same as X_temp
                        # if X_temp.shape[0] == temp.shape[0]:
                        X_temp = np.hstack([X_temp, temp[5]])  # Append temp horizontally to X_temp
                        # else:
                        #     # Handle mismatch in the number of rows (e.g., raise an error or skip the temp array)
                        print("Mismatch in the number of rows. Skipping this temp array.")


                else:
                    print("Distance from origin 4")
                    imported_table = pd.read_excel(active_file_data, sheet_name=active_sheet, usecols="A:C")
                    temp = imported_table.to_numpy()
                    # X_temp = np.hstack([X_temp, temp[5]])
                    # X_temp = temp[0]

            else:
                imported_table = pd.read_excel(active_file_data, sheet_name=active_sheet, usecols="A:A")
                temp = imported_table.to_numpy()
                
               
                if X_temp.shape[0] == 0:
                    X_temp = temp

                else:
                    # Check if the number of rows in temp is the same as X_temp
                    if X_temp.shape[0] == temp.shape[0]:
                        X_temp = np.hstack([X_temp, temp])  # Append temp horizontally to X_temp
                    else:
                        # Handle mismatch in the number of rows (e.g., raise an error or skip the temp array)
                        print("Mismatch in the number of rows. Skipping this temp array.")

              
                file_length = len(temp)
        

        
    

        
      
                
        active_file_data.close

        

        X_temp = X_temp[1:]
        num_rows = X_temp.shape[0]

        try:
            y_temp = np.array([file.cellline.name for _ in range(num_rows)])
            y = np.hstack([y, y_temp])
        except:
            y= np.array([file.cellline.name for _ in range(num_rows)]) 
        

    
        try:
            print("trying")    
            X = np.concatenate((X, X_temp), axis=0)
            
            # X.append(X_temp)
            print("succeeding")
        except:
            
            X = X_temp 
        # Example dimensions check




    X = np.vstack(X)  # Convert the list of arrays into a single NumPy array
  
    # X_labels = np.vstack(X_labels).flatten()


    volume_idx = np.flatnonzero(param_labels == "Volume")  # Correct the param_labels check
    mask = X[:, volume_idx] >= 0
    X_new = X[mask]
    
    # shutil.rmtree(temp_dir)
    # return X
    
    train(X,y)
    return X_new, X_labels,  cell_list

# Function 2: parameter_calc
def parameter_calc(X, objectID):
    

    labels = X[0].tolist()
    figures = X[1:].tolist()
     
    area= np.array([row[labels.index('Area')] for row in figures])
    volume = np.array([row[labels.index('Volume')] for row in figures])  # bbaa_x = [row[labels.index('BB AA X')] for row in figures]
    # bbaa_z = [row[labels.index('BB AA Z')] for row in figures]
    # bboo_x = [row[labels.index('BB OO X')] for row in figures]
    # bboo_z = [row[labels.index('BB OO Z')] for row in figures]
    # pos_x =  [row[labels.index('Position X')] for row in figures]
    # pos_y =  [row[labels.index('Position Y')] for row in figures]
    # pos_z =  [row[labels.index('Position Z')] for row in figures]


    # area = X[:, np.where("param_labels" == "Area")]
    # volume = X[:, np.where("param_labels" == "Volume")]
    bbaa_x = X[:, np.where("param_labels" == "BB AA X")]
    bbaa_z = X[:, np.where("param_labels" == "BB AA Z")]
    bboo_x = X[:, np.where("param_labels" == "BB OO X")]
    bboo_z = X[:, np.where("param_labels" == "BB OO Z")]
    pos_x = X[:, np.where("param_labels" == "Position X")]
    pos_y = X[:, np.where("param_labels" == "Position Y")]
    pos_z = X[:, np.where("param_labels" == "Position Z")]

    ci = (area ** 3) / (16 * (np.pi ** 2) * (volume ** 2))
    mbi_aa = bbaa_x / bbaa_z
    mbi_oo = bboo_x / bboo_z
    polarity = np.arctan2(pos_y, np.sqrt(pos_x ** 2 + pos_z ** 2))

    # Check dimensions of ci, mbi_aa, mbi_oo, and polarity
    # and make sure they have the same number of rows as X
    # before concatenating.
    

    # If all have the same number of rows as X, you can concatenate them.
    X_3d = X[:, np.newaxis, :]
    
    print(X_3d.shape, ci.shape, mbi_aa.shape, mbi_oo.shape, polarity.shape)
    # X_new = np.hstack([X_3d, ci, mbi_aa, mbi_oo, polarity])


    cellnum = np.sum(objectID == 0)
    mmdist_data = []

    start_index = np.flatnonzero(objectID == 0)
    for i in range(cellnum):
        cellstart = start_index[i]
        cellend = start_index[i + 1] if i + 1 < cellnum else len(objectID) - 1
        cellrange = np.arange(cellstart, cellend + 1)

        current_mmdist_data = np.zeros((len(cellrange), len(cellrange)))

        for x in range(len(cellrange)):
            for y in range(len(cellrange)):
                if x == y:
                    current_mmdist_data[x, y] = 0
                    continue

                obj1_idx = cellrange[x]
                obj2_idx = cellrange[y]
                # obj1_pos = X[obj1_idx, [np.where("param_labels" == "Position X")[0][0],
                #                         np.where("param_labels" == "Position Y")[0][0],
                #                         np.where("param_labels" == "Position Z")[0][0]]]
                # obj2_pos = X[obj2_idx, [np.where("param_labels" == "Position X")[0][0],
                #                         np.where("param_labels" == "Position Y")[0][0],
                #                         np.where("param_labels" == "Position Z")[0][0]]]

                obj1_pos =[16,15,5]
                # figures[obj1_idx, [
                #                         # np.array([row[labels.index('Area')] for row in figures]),
                #                         # np.array([row[labels.index('Area')] for row in figures]),
                #                         # np.array([row[labels.index('Area')] for row in figures])
                #                         12,23,54
                #                         ]]
                obj2_pos = [16,15,5]
                # figures[obj2_idx, [
                #                         # np.array([row[labels.index('Area')] for row in figures]),
                #                         # np.array([row[labels.index('Area')] for row in figures]),
                #                         # np.array([row[labels.index('Area')] for row in figures])
                #                         23,45,53
                # ]]

                mm_dist = np.sqrt((obj1_pos[0] - obj2_pos[0]) ** 2 +
                                  (obj1_pos[1] - obj2_pos[1]) ** 2 +
                                  (obj1_pos[2] - obj2_pos[2]) ** 2)

                current_mmdist_data[x, y] = mm_dist


        mmdist_data.append(current_mmdist_data)

    min_mmdist = []
    max_mmdist = []
    mean_mmdist = []
    median_mmdist = []
