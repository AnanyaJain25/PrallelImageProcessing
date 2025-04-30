CUDA Image Processing Project
This project implements a CUDA-based image processing application to perform four operations—Brightness Adjustment, Grayscale Conversion, Blurring, and Sharpening—on PNG images. The goal is to leverage NVIDIA GPU parallelization to accelerate these operations compared to their serial CPU implementations. The program measures execution times, generates output images, and visualizes performance using plots.
Features

Operations:
Brightness Adjustment: Increases pixel intensity by a constant.
Grayscale Conversion: Converts RGB images to grayscale using the luminance formula (0.299R + 0.587G + 0.114B).
Blurring: Applies a 1D averaging filter (3-pixel neighborhood).
Sharpening: Enhances edges using a sharpening filter.


Parallelization: CUDA kernels for each operation, executed on a Tesla T4 GPU.
Performance Comparison: Serial (CPU) and parallel (GPU) execution times for all operations.
Output: Eight PNG images (serial and parallel outputs for each operation) and execution time plots.
Environment: Developed and tested in Google Colab with CUDA 11.8.

Performance Results
The program was tested on a 512x512 RGB image using a Tesla T4 GPU, yielding the following execution times:
Brightness Serial: 106.26 ms
Brightness Parallel: 1.20 ms
Grayscale Serial: 92.72 ms
Grayscale Parallel: 2.11 ms
Blurring Serial: 190.42 ms
Blurring Parallel: 1.18 ms
Sharpening Serial: 190.30 ms
Sharpening Parallel: 1.18 ms


Speedups: CUDA implementations achieved speedups of 43.9x to 161.4x over serial implementations.
Plots:
execution_time_comparison.png: Bar chart comparing serial (red) and parallel (green) times.
speedup_comparison.png: Bar chart showing speedup (Serial / Parallel).



Prerequisites

Hardware: NVIDIA GPU (e.g., Tesla T4, Compute Capability 7.5 or higher).
Software:
CUDA Toolkit 11.8
libpng (for PNG I/O)
Python 3 (for plotting)
Matplotlib (for visualization)


Optional: Google Colab with GPU runtime for easy setup.

Setup Instructions
1. Clone the Repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

2. Set Up the Environment
On Google Colab (Recommended)

Open a new Colab notebook.
Enable GPU runtime: Runtime > Change runtime type > Select GPU > Save.
Install dependencies:!apt-get update
!apt-get install -y libpng-dev imagemagick
!wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
!chmod +x cuda_11.8.0_520.61.05_linux.run
!./cuda_11.8.0_520.61.05_linux.run --silent --toolkit --installpath=/usr/local/cuda-11.8
!export PATH=/usr/local/cuda-11.8/bin:$PATH
!export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH


Install Python dependencies:!pip install matplotlib



On a Local Machine

Install CUDA Toolkit 11.8: Follow NVIDIA’s installation guide.
Install libpng:
Ubuntu: sudo apt-get install libpng-dev
macOS: brew install libpng


Install Python and Matplotlib:pip install matplotlib



3. Prepare an Input Image

Use a PNG image (e.g., 512x512 pixels). Example:wget https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg -O image.jpg
convert image.jpg image.png



Usage
1. Compile the Program
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
/usr/local/cuda-11.8/bin/nvcc -o job_parser job_parser.cu -lpng -O3 -arch=compute_60 -code=sm_60,sm_75,sm_80

2. Run the Program
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH ./job_parser image.png


Outputs:
Eight PNG images: brightness_out.png, cuda_brightness_out.png, grayscale_out.png, cuda_grayscale_out.png, blur_out.png, cuda_blur_out.png, sharpen_out.png, cuda_sharpen_out.png.
execution_times.txt: Execution times for all operations.



3. Generate Plots
python3 plot_execution_times.py


Outputs:
execution_time_comparison.png
speedup_comparison.png



Project Structure

job_parser.cu: Main CUDA program implementing serial and parallel image processing operations.
plot_execution_times.py: Python script to generate execution time and speedup plots.
execution_times.txt: Output file containing execution times.
*.png: Output images and plots.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch: git checkout -b feature/your-feature.
Make your changes and commit: git commit -m "Add your feature".
Push to your branch: git push origin feature/your-feature.
Open a pull request.

