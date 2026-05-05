# weights/

Place your trained weights file here as:

    weights/tcn_model.pt

Or pass an explicit path to predict_streaming():

    predict_streaming(imu_window, weights_path='/path/to/subjectXXX.pt')

If running LOSOCV models (one .pt per subject), pass the appropriate
subject-specific weights file when calling predict_streaming().
