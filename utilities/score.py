#!/usr/bin/env python

# Scoring program for the See4C challenge
# Isabelle Guyon, January 2016

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, SEE4C, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

# Some libraries and options
import os

from sys import argv, path
import sys
import numpy as np
from metric import scoring_function, score_name
from libscores import ls, filesep, mkdir, mvmean, swrite
from libscores import show_platform, show_io, show_version
from data_manager import DataManager
from time import time
from data_io import vprint

# Default I/O directories:      
root_dir = "../.."
default_code_dir = os.path.join("../")
default_data_dir = os.path.join(default_code_dir, "sample_data")
default_solution_dir = os.path.join(default_data_dir, 'adapt')
default_result_dir = os.path.join(default_code_dir, "results") 
default_cache_dir = os.path.join(default_code_dir, "cache") 
default_output_dir = os.path.join(root_dir, "scoring_output") 

# Debug flag 0: no debug, 1: show all scores, 2: chaeting results (use ground truth)
debug_mode = 1
verbose = True

# Constant used for a missing score
missing_score = 0.999999

# Version number
scoring_version = 1.0

# Extension of the files
ext = '.h5'

# Number of frames that are expected to be predicted
frame_num = 11
    
# =============================== MAIN ========================================
    
if __name__=="__main__":

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = root_dir
        data_dir = default_data_dir
        code_dir = default_code_dir
        solution_dir = default_solution_dir 
        result_dir = default_result_dir 
        cache_dir = default_cache_dir
        output_dir = default_output_dir
        
    elif len(argv)==3:
        input_dir = argv[1]
        output_dir = argv[2] 
        # Indentify various directories and create in necessary
        data_dir = os.path.join(input_dir, 'ref') # Contains subdirectories train/ and adapt/
        code_dir = os.path.join(input_dir, 'res')
        solution_dir = os.path.join(data_dir, 'adapt')
        result_dir = os.path.join(code_dir, 'results')
        cache_dir = os.path.join(code_dir, 'cache')
        
    else:
        data_dir = argv[1]
        output_dir = argv[2]  
        code_dir = argv[3] 
        solution_dir = os.path.join(data_dir, 'adapt')
        result_dir = os.path.join(code_dir, 'results')
        cache_dir = os.path.join(code_dir, 'cache')  	
    
    mkdir(output_dir) 
    mkdir(cache_dir) 

    
    
    # Get the prediction program from the user
    # This should be a file called predictSpatioTemporal.py in the res directory
    # res/ should also contain a subdirectory wirh all code dependencies
    # print path
    # print os.path.dirname(os.path.abspath(__file__))
    # print os.getcwd()
    # os.chdir('/Users/lishengsun/Downloads/zSee4C_starting_kit_Lisheng')
    # print os.listdir(os.getcwd())
    # print os.path.abspath(root_dir)
    # print len(argv)
    sys.path.append(code_dir)

    # path.append('/Users/lishengsun/Downloads/zSee4C_starting_kit')
    from predictSpatioTemporal import predictSpatioTemporal as predict

    # Create the output directory, if it does not already exist and open output files  
    score_file = open(os.path.join(output_dir, 'scores.txt'), 'wb')
    html_file = open(os.path.join(output_dir, 'scores.html'), 'wb')
    
    # Get all the solution files from the solution directory
    solution_names = ls(os.path.join(solution_dir,'*' + ext))
    steps = [int(S[-S[::-1].index(filesep):-S[::-1].index('.')-1][1:]) for S in solution_names]
    if not steps: raise IOError('No solution file found in {}'.format(solution_dir))
    steps = np.sort(steps)
    max_steps = len(steps)-1
    assert np.all(steps==range(max_steps+1))
    
    # max_steps = 1
    
    # Compute predictions (except on the last file)
    start_time = time()         # <== Mark starting time
    for step_num in range(max_steps):
        if True: #try:
            predict(step_num, data_dir, result_dir, code_dir, cache_dir=cache_dir, num_predicted_frames=frame_num)
        if False: #except:
            raise Exception('Error in prediction program') 
    time_spent = time() - start_time
    
    # Score results (except on the first file)
    score = np.zeros(max_steps)
    for step_num in range(max_steps): 
        sfile = 'X' + str(step_num+1)
        pfile = 'Y' + str(step_num)
        vprint( verbose,  "************************************************")
        vprint( verbose,  "******** Predictions=%s; Truth=%s ************" % (pfile, sfile))
        vprint( verbose,  "************************************************")
        # Extract the step number from the file name
        try:
            # Get the solution from the res subdirectory (must end with ext)
            solution_file = os.path.join(solution_dir, sfile + ext)
            if debug_mode==2:
                predict_file = solution_file
            else:
                predict_file = os.path.join(result_dir, pfile + ext)
            if (predict_file == []): raise IOError('Missing prediction file in step {}'.format(step_num))

            # Read the solution and prediction values into numpy arrays
            Dsolution = DataManager(datatype="solution", data_file=solution_file, verbose=verbose, two_d_map=False)
            # print Dsolution.t.shape
            # print frame_num
            Dprediction = DataManager(datatype="prediction", data_file=predict_file, verbose=True, two_d_map=False) # already shape(time,44,44)
            # Dprediction.map_back_to_1d(Dprediction.X)
            
            if Dprediction.X.shape[0] != frame_num: 
                vprint(verbose, 'WARNING: Wrong number of predicted frames {}'.format(Dprediction.X.shape[0]))
                vprint(verbose, '         Keeping only the first {}'.format(frame_num))
                Dprediction.X = Dprediction.X[0:frame_num]
            prediction = Dprediction.X.ravel()
            if Dsolution.X.shape[0] != frame_num: 
                vprint(verbose, 'WARNING: Wrong number of solution frames {}'.format(Dsolution.X.shape[0]))
                vprint(verbose, '         Keeping only the first {}'.format(frame_num))   
                Dsolution.X = Dsolution.X[0:frame_num]
            solution = Dsolution.X.ravel()
            

            if(solution.shape!=prediction.shape): raise ValueError("Bad prediction shape {}".format(prediction.shape))

            try:
                # Compute the score prescribed by the metric file 
                score[step_num] = scoring_function(solution, prediction)
                print("======= Step %d" % step_num + ": score(" + score_name + ")=%0.12f =======" % score[step_num])                
                html_file.write("======= Step %d (%s,%s)" % (step_num, pfile, sfile) + ": score(" + score_name + ")=%0.12f =======\n" % score[step_num])
            except:
                raise Exception('Error in calculation of the specific score of the task')
                                    
        except Exception as inst:
            print inst
            score = missing_score 
            print("======= Step %d" % step_num + ": score(" + score_name + ")=ERROR =======")
            html_file.write("======= Step %d" % step_num + ": score(" + score_name + ")=ERROR =======\n")            

    # End loop for solution_files
        
    # Average scores and ** take square root **
    RMSE = np.sqrt(mvmean(score))
    print("*** RMSE = %0.12f ***" % RMSE) 
    html_file.write("*** RMSE = %0.12f ***" % RMSE)
    # Write score corresponding to selected task and metric to the output file
    if np.isnan(RMSE): RMSE=missing_score
    score_file.write("RMSE: %0.12f\n" % RMSE)
    # Record the execution time and add it to the scores
    score_file.write("Duration: %0.6f\n" % time_spent)
    		
    html_file.close()
    score_file.close()

    # Lots of debug stuff
    if debug_mode>1:
        swrite('\n*** SCORING PROGRAM: PLATFORM SPECIFICATIONS ***\n\n')
        show_platform()
        show_io(input_dir, output_dir)
        show_version(scoring_version)
		
    #exit(0)



