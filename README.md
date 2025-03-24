# Memory Footprint Test
This application is designed to test memory usage during model compilation and inference using OpenVINO.

# Notes
- During the first run of this application, cache blob files will be generated.
- When using the `AUTO` device, CPU loading will be disabled at second run to improve performance. Memory footprint measurements will be available from the second run.

# How to build
```bash
cd memory_footprint
mkdir build && cd build
source /path/to/OpenVINO/folder/setupvars.sh
cmake ..
make
```
# How to run
## 1. Single model compilation and inference
```bash
./main <Target Device> </path/to/model.xml>
```
Example:
```bash
    ./main AUTO:NPU,CPU /path/to/model.xml
```

## 2. Multiple models compilation and inference on first model
```bash
./main <Target Device> </path/to/model_1.xml>  </path/to/model_2.xml> </path/to/model_3.xml>
```
Example:
```bash
    ./main AUTO:NPU,CPU /path/to/model_1.xml /path/to/model_2.xml /path/to/model_3.xml
```