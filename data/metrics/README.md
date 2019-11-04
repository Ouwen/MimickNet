This file includes metrics ran for MimickNet across our test and prospective datasets.

 - Graybox refers to the gray-box training scheme
 - MimickNet refers to the black-box training scheme
 - Pro refers to prospective data used for evaluation only
 - No clip refers to metrics collected on data that is not clipped to a -80dB noise floor
 - Cardiac refers to a cardiac cine loop used for evaluation only
 - REFocUS refers to REFocUS beamforming
 - Gain Corrected refers to adding a mean gain offset to the image

Within each CSV,
 - hm prefix refers to histogram matched images where clinical-grade post-processing and MimickNet images are matched to DAS.
 - iq prefix refers to DAS
