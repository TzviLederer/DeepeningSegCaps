# DeepeningSegCaps
Code of the paper Deepening SegCaps  

## Project Structure
segcaps contains the training and testing code of segCaps model for SCR data- with only heart label  
segcaps_CHAOS contains the training and testing code of segCaps model for CHAOS data  
The file segcaps_CHAOS\capsule_layers.py contains our change of the original segCaps code to follow the paper description  
segcaps_multilabel contains the training and testing code of segCaps multilabel extension model for SCR data  

## Data Structure
Inside the data root folder (i.e. where the data stored)  should be two folders:  
one called imgs and one called masks.  
All models, results, etc. are saved to this same root directory.

## Train 
To train the model put the data in the correponding directory in the data directory.

## Test
The weights are too large to upload to github.  Please contact us for it:  
Tzvi Lederer: tzvilederer@mail.tau.ac.il  
Shira Kasten-Selin: kastenserlin@mail.tau.ac.il
