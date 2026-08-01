# RT-FAREP

Real-Time Fog-Adaptive Face Recognition Pipeline.

Extension of FAREP (FYP-1) into a real-time, deployable system with TensorRT optimization.

## Structure
- configs/ — hyperparameters and paths
- data/ — dataset download and fog synthesis scripts
- models/ — model architectures (fog classifier, GAN, ArcFace)
- training/ — training loops
- inference/ — real-time pipeline
- evaluation/ — metrics (PSNR, SSIM, TAR/FAR)
- notebooks/ — Kaggle training notebooks
- demo/ — webcam demo app
