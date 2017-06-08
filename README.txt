This directory contains all you should need to prepare a sample submission for the See.4C RTE challenge.

This code was tested with 
Python 2.7.13 | Anaconda 4.3.1 (https://anaconda.org/)
Keras 1.2.1 (https://keras.io/)
Tensorflow 0.11.0rc2 (follow Keras instructions)

ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". The SEE.4C CONSORTIUM, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. IN NO EVENT SHALL AUTHORS AND ORGANIZERS BE LIABLE FOR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.

On the See.4C server, the script predict.sh is called. This is the main entry point, which should be used as an interface if you do not use Python or do not follow our Python interface. After calling your code is called the makeready.sh script, which make available the next test video.

If you follow our Python interface, use predictSpatioTemporal.py as your main entry point (it is called by predict.sh)

Usage: 
=====

	python predictSpatioTemporal.py 0 sample_data results `pwd`


Change 0 to 1, 2, etc. to more to other steps.
By default predictSpatioTemporal.py is a copy of predictSpatioTemporal/predictSpatioTemporal-persistence.py. See other examples in predictSpatioTemporal/.

IMPORTANT:
=========
1) DATA — Directory “sample_data” contains a mini-dataset for debug purposes. Replace sample_data by public_data, if you want a larger dataset, in the above command. On the server, you will have available a (different) dataset in a similar format. More training data will be available in the train/, in subdirectories called Xmn, where n is an integer number.
2) CODE — your code must be called predictSpatioTemporal.py. You will find several examples of models in the predictSpatioTemporal/ directory. Under Unix (Linux, Ubuntu, etc.) you can just copy the one you want to try to replace predictSpatioTemporal.py.
For the two most basic examples (persistence and linear kernels), the code is in fact the same, the only difference is what prediction mode is loaded:
from data_manager import DataManager # load/save data and get info about them
**************** CHANGE THIS **************** 
from simple_model import Model 
3) VISUALIZATION — we also provide code to explore/visualize the data, try:
	jupyter-notebook README.ipynb
4) SCORING — we supply a scoring program that allows you to compute your RMSE on sample_data or public_data, try:
	python utilities/score.py sample_data results `pwd`

Note: we use the “cache” directory to save intermediary results. The predictions are saved in the results directory.
