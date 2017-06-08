This directory examples of predictive models for the See.4C video forecasting contest.

ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". The SEE.4C CONSORTIUM, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. IN NO EVENT SHALL AUTHORS AND ORGANIZERS BE LIABLE FOR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.

You must copy predictSpatioTemporal_xxx.py to ../predictSpatioTemporal.py

1) predictSpatioTemporal_persistence.py: The most basic (and fast) model. it just predicts that the next farm is identical to the last one. Unfortunately this “stupid” model is hard to beat!
2) predictSpatioTemporal_linear.py: A slightly fancier auto-regressive linear model.
3) predictSpatioTemporal_prednet.py: A deep learning method.
RIGHT NOW THIS MODEL DOES NOT GET TRAINED (because training is very slow unless you have a GPU). To make it train, set DEBUG_MODE = 0.





