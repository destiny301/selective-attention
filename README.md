# Improving the Efficiency of CMOS Image Sensors through In-Sensor Selective Attention ([ISCAS2023](https://ieeexplore.ieee.org/abstract/document/10181835/))

## Abstract
Inspired by the selective attention mechanism in human vision, we propose to introduce a saliency-based processing step in the CMOS image sensor, to continuously select pixels corresponding to salient objects and feedback such information to the sensor, instead of blindly passing all pixels to the sensor output. To minimize the overhead of saliency detection in this feedback loop, we propose two techniques: (1) saliency detection with low-precision, down-sampled grayscale images, and (2) Optimization of the loss function and model structure. Finally, we pad the minimum number of pixels around the selected pixels to maintain the accuracy of object detection (OD). Our method is experimented with two types of OD algorithms on three representative datasets. At the similar OD accuracy with the full image, our proposed selective feedback method successfully achieves 70.5% reduction in the volume of output pixels for BDD100K, which translates to 4.3× and 3.4× reduction in power consumption and latency, respectively.

![image](https://github.com/destiny301/selective-attention/blob/main/figures/intro.jpg)

It works for different datasets, like MSRA10K, BDD100K, and COCO2017, as the following flowchart.

![image](https://github.com/destiny301/selective-attention/blob/main/figures/flowchart.jpg)

## Citation
If you use DPR in your research or wish to refer to the results published here, please use the following BibTeX entry. Sincerely appreciate it!
```shell
@inproceedings{zhang2023improving,
  title={Improving the Efficiency of CMOS Image Sensors through In-Sensor Selective Attention},
  author={Zhang, Tianyi and Kasichainula, Kishore and Jee, Dong-Woo and Yeo, Injune and Zhuo, Yaoxin and Li, Baoxin and Seo, Jae-sun and Cao, Yu},
  booktitle={2023 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages={1--4},
  year={2023},
  organization={IEEE}
}
```
