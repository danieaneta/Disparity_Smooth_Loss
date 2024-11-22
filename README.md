

Disparity Smoothness Loss is a loss function showcased in the research paper "Unsupervised Monocular Depth Estimation with Left-Right Consistency" by University College of London. 
https://arxiv.org/pdf/1609.03677


This repo is exploring this loss function from paper to code.

Included in this repo:
- My notes and research
- Pixel Logic Exploration
- Code Implementation
- Debugging History
- Optimization History

___

<h3> Hardware Testing: </h3>

Intel i9-13900H (13th Gen) - ~2.6GHz
65MB RAM 
GPU: NVIDIA GeForce RTX 4050 Laptop GPU


| Version       | V01_Logic_Testing | V02_Logic_Testing | V03_dataclass_simple_numba_numpy | V04_half_torch_numpy | V05_full_numpy_scipy | V06_full_pytorch  |
| ------------- |:-----------------:|:-----------------:|:--------------------------------:|:--------------------:|:--------------------:|:-----------------:|
| Total Processing Time R1      | 0:17              | 2:41              | 1:19                             | 0:07                 | 0:00.75              |0:02              |
| Total Processing Time R2     | 0:16              | 2:48              | 1:19                             | 0:09                 | 0:00.45              |0:02              |
| Total Processing Time R3     | 0:17              | 2:42              | 1:20                             | 0:09                 | 0:00.83              |0:01              |
| Third-Party Optimization Percentage     | 0%              | 9.6%(Numpy)              | 3.7%(Numpy), 16.9%(Numba), 2.8%(PyTorch)                             | 0.19% (Numpy), 7.7%(PyTorch)                 | 100%              | 100%             |


Intel i7-12700 (12th Gen) - ~2.1GHz
32GB RAM
GPU: NVIDIA GeForce RTX 3070ti

| Version       | V01_Logic_Testing | V02_Logic_Testing | V03_dataclass_simple_numba_numpy | V04_half_torch_numpy | V05_full_numpy_scipy | V06_full_pytorch  |
| ------------- |:-----------------:|:-----------------:|:--------------------------------:|:--------------------:|:--------------------:|:-----------------:|
| Total Processing Time R1      | 0:19              | 2:48              | 1:31                             | 0:13                 | 0:01              |0:03              |
| Total Processing Time R2     | 0:18              | 2:45              | 1:33                             | 0:12                 | 0:00.88              |0:03              |
| Total Processing Time R3     | 0:19              | 2:44              | 1:33                             | 0:13                 | 0:01              |0:03              |
| Third-Party Optimization Percentage     | 0%              | 9.6%(Numpy)              | 3.7%(Numpy), 16.9%(Numba), 2.8%(PyTorch)                             | 0.19% (Numpy), 7.7%(PyTorch)                 | 100%              | 100%             |

___

<h3>Explanations: </h3>

**V01_Logic_Testing:** Written in Python without any third-party optimization libraries. No optimization and vectorization. Going through the multi-step equation of disparity smooth loss for research purposes and validate equation steps and translation to Python code. The average time of this version is used as the base time, where any time more than this average time is considered as bloat and time below is seen as acceptable/exporable. 

Best Time: 0:16
Worst Time: 0:19

**V02_Logic_Testing:** Handling the original Python logic with numpy arrays. Minimal Numpy integration which has coordinates stored into numpy arrays. This causes "bloat time" in processing. Original logic implementation is faster only because of straightforward indexing and subtraction operations. The creations of small numpy arrays for every calculation of the difference creates overhead. "Refactoring" code in this way would be the least cost-effective process and would lead to issues immediately and down the line.

Best Time: 2:41
Worst Time: 2:48

**V03_dataclass_simple_numba_numpy:**

Best Time: 1:19
Worst Time: 1:33

**V04_half_torch_numpy:**

Best Time: 0:07
Worst Time: 0:13

**V05_full_numpy_scipy:**

Best Time: 0:00.45
Worst Time: 

<h3>Research Notes:</h3>
![Screenshot 2024-11-12 154714](https://github.com/user-attachments/assets/37de225b-2721-442b-8b36-a329a0040fa4)
![Screenshot 2024-11-12 154737](https://github.com/user-attachments/assets/6951d0c8-3905-47d2-a0a4-6055788dde39)
![Screenshot 2024-11-12 154805](https://github.com/user-attachments/assets/32c13086-8937-4dcf-92f8-2212f6197d9a)
