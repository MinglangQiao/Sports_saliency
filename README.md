# Sports_saliency

This repository provides the database and code for reproducing the results in the paper: 

* [**Saliency Prediction of Sports Videos: A Large-scale Database and a Self-Adaptive Approach.**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10446481)
[*Minglang Qiao*](http://45.77.201.133/html/Members/minglangqiao.html),
[*Mai Xu*](XX),
[*Shijie Wen*](XX),
[*Lai Jiang*](XX),
[*Shengxi Li*](XX),
[*Tao Xu*](XX),
[*Yunjin Chen*](XX),
[*Leonid Sigal*](XX).

Accepted by ICASSP 2024.


# Database

![image](https://github.com/MinglangQiao/Sports_saliency/blob/main/%E8%AE%BA%E6%96%87%E7%89%88%E6%9C%AC.png)


The database can be download from [Here](https://www.dropbox.com/scl/fi/qo6eyp6mgoayfzk5xo0mn/sports_database.zip?rlkey=i77bzy60gmf72xwu8k2ahx8se&st=lileu1mt&dl=0). Due to the copyright of the videos, the database is password protected, 
please send me email (minglangqiao@buaa.edu.cn) to get the password of the download link. 


# Test Code

```
python test_single_video.py --gpu 6 --use_sound True --residual_fusion True \
    --test_video_path "/xx/all_1000video/out_of_play_(2).mp4" \
    --save_path "/xx/model_output/" \
    --file_weight "/xx/VSTNet_pseudo_test.pth"
```

The pre-trained model can be download from [Here](https://www.dropbox.com/scl/fi/y0b885yw5z89x9fhrmlha/VSTNet_pseudo_test.pth?rlkey=6bq275lrieuf1ruo1u6ri4kxm&st=um7rpxsk&dl=0).


# Training Code

TO BE UPDATE

