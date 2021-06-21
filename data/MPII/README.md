# MPII Dataset

* **[Website](http://human-pose.mpi-inf.mpg.de/#)**
* Andriluka, Mykhaylo, et al. "2d human pose estimation: New benchmark and state of the art analysis." Proceedings of the IEEE Conference on computer Vision and Pattern Recognition. 2014. [**[paper](https://ieeexplore.ieee.org/document/6909866)**][**[pdf](https://openaccess.thecvf.com/content_cvpr_2014/papers/Andriluka_2D_Human_Pose_2014_CVPR_paper.pdf)**]

## How to download?
The dataset is about **12.1 GB**. In case you prefer to download it on your computer, you can use your terminal for it.
Execute the following command in your terminal to download the whole dataset.

```bash
cd <torch_shnet path>/data/MPII/
curl -o mpii_human_pose_v1.tar.gz https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
tar -xvzf mpii_human_pose_v1.tar.gz
```