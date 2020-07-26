## Python3
sudo apt-get update
sudo apt-get install python3
## PyQt5 (for Python3)
sudo apt-get install python3-pyqt5

## PyDicom
python3 -m pip install pydicom

## Dependencies
sudo apt-get install -y python3-tk
python3 -m pip install --upgrade pip
python3 -m pip install matplotlib scipy scikit-image numpy opencv-python

python3 -m pip install --no-cache-dir git+https://github.com/pydicom/pydicom.git


# Extra dependencies for PyDicom
# OpenSSL
sudo apt-get install libssl-dev
# Install latest CMAKE package. At least, CMake 3.9.2. For now, compiling from source is required.
## Step 1: Download and uncompress the source
## Step 2: Remove APT package:
sudo apt-get remove cmake && sudo apt-get purge cmake
## Step 3: Run
./bootstrap && make && sudo make install

# Install SWIG
sudo apt-get install swig

# Install GDCM
http://gdcm.sourceforge.net/wiki/index.php/Configuring_and_Building

On UNIX (with cmake) this is simply a matter of doing:
 * git clone --branch release git://git.code.sf.net/p/gdcm/gdcm
 * mkdir gdcmbin
 * cd gdcmbin
 * cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/gdcm -DGDCM_WRAP_PYTHON=true "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=true"
   [select your configuration]
 * Press 'c' (configure), Press 'g' (generate)
 * make
The following step is not required, as gdcm will work from a build tree too:
 * sudo make install

Major options explained:
* GDCM_BUILD_SHARED_LIBS: Turn it on if you want shared libs (instead of static libs), greatly reduce executable size, allowing code reuse.
* GDCM_WRAP_PYTHON: turn it on if you want to be able to access the GDCM API via python (required python dev files)
* GDCM_WRAP_CSHARP: turn it on if you want to be able to access the GDCM API via C# (required mono or .NET environment)
* GDCM_WRAP_JAVA: turn it on if you want to be able to access the GDCM API via java (required java sdk to compile)
* GDCM_WRAP_PHP: turn it on if you want to be able to access the GDCM API via php (experimental)
* GDCM_USE_VTK: turn if on if you want to be able to load DICOM file in VTK context (requires VTK)
* GDCM_BUILD_APPLICATIONS: turn it on if you want the build gdcm applications (gdcmdump, gdcmconv, gdcminfo ...)
* GDCM_BUILD_TESTING: Turn it on if you want to be able to exectute GDCM testing suite
* GDCM_DOCUMENTATION: turn it on if you want to generate the developer documentation (require doxygen)
* GDCM_BUILD_EXAMPLES: turn it on if you want to build simple examples that demonstrates GDCM usage.