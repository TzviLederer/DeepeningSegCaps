# DeepeningSegCaps
Code of the paper Deepening SegCaps
segcaps contains the training and testing code of segCaps model for SCR data- with only heart label
segcaps_CHAOS contains the training and testing code of segCaps model for CHAOS data
The file segcaps_CHAOS\capsule_layers.py contains our change of the original segCaps code to follow the paper description
segcaps_multilabel contains the training and testing code of segCaps multilabel extension model for SCR data

Dataset Structure
Inside the data root folder (i.e. where the data stored)  should be two folders: one called imgs and one called masks. All models, results, etc. are saved to this same root directory.