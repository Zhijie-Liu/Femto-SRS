# Femto-SRS
Machine learning project for gastric cancer diagnosis in our paper titled "Instant diagnosis of gastroscopic biopsy via deep-learned single-shot femtosecond stimulated Raman histology" published in Nature Communications.

For U-Net: 

Run the files "Unet_two_batchnorm.py" and "Unet_two_gastric_batchnorm.py" to map the HeLa cell and gastric tissue respectively and obtain the network file, and then use the corresponding "test" file to generate the mapping conversion image.

For CNN:

Run the files "Inception_Resnet_v2_twoclass.py" and "Inception_Resnet_v2_low_to_high.py" to train the network from cancer-noncancer datasets and diff.-undiff. datasets, then use the corresponding "check" file to obtain accuracy in testsets.

Please contact the corresponding author to get the whole data sets.
