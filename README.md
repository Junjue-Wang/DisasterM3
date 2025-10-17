<h2 align="center">
  <img
    src="https://github.com/Junjue-Wang/resources/blob/main/DisasterM3/icon.png?raw=true"
    alt="Disaster icon"
    height="50"
    style="vertical-align:-16px;"
  />
  DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response
</h2>

<h5 align="center"><a href="https://junjue-wang.github.io/homepage/">Junjue Wang*</a>,
<a href="https://weihaoxuan.com">Weihao Xuan*</a>,
Heli Qi, Zhihao Liu, Kunyi Liu, Yuhan Wu, <a href="https://chrx97.com/"> Hongruixuan Chen</a>,
<a href="https://jtrneo.github.io/"> Jian Song</a></h5>
<h5 align="center">
Junshi Xia, <a href="https://zhuozheng.top/">Zhuo Zheng</a>, <a href="https://naotoyokoya.com/">Naoto Yokoya†</a></h5>

<h5 align="center">
* Equal Contributions
† Corresponding Author</h5>

[[`Paper`](https://arxiv.org/abs/2505.21089)],
[[`Dataset`](https://forms.gle/APQpmyuThh28HsJdA)]


<div align="center">
  <img src="https://github.com/Junjue-Wang/resources/blob/main/DisasterM3/task_taxonomy.png?raw=true">
</div>

## Highlights
DisasterM3 includes 26,988 bi-temporal satellite images and 123k instruction pairs across 5 continents, with three characteristics:
1. Multi-hazard: 36 historical disaster events with significant impacts, which are categorized into 10 common natural and man-made disasters
2. Multi-sensor: Extreme weather during disasters often hinders optical sensor imaging, making it necessary to combine Synthetic Aperture Radar (SAR) imagery for post-disaster scenes
3. Multi-task: 9 disaster-related visual perception and reasoning tasks, harnessing the full potential of VLM's reasoning ability


## News
- 2025/09/22, We are preparing the dataset and code.
- 2025/09/22, Our paper got accepted by NeurIPS 2025.


## Benchmark

Please run this code for benchmarking the DisasterM3 dataset.
Two examples:
Qwen2.5 VL:
```
python disaster_m3/pyscripts/run_vllm.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --subset bearing_body
```
InternVL3:
```
python disaster_m3/pyscripts/run_vllm.py --model_id OpenGVLab/InternVL3-78B --subset report
```


## Citation
If you use DisasterM3 in your research, please cite our following papers.
```text
  @article{wang2025disasterm3,
  title={DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response},
  author={Wang, Junjue and Xuan, Weihao and Qi, Heli and Liu, Zhihao and Liu, Kunyi and Wu, Yuhan and Chen, Hongruixuan and Song, Jian and Xia, Junshi and Zheng, Zhuo and Yokoya, Naoto},
  booktitle={Proceedings of the Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgments
This dataset builds upon the following excellent open datasets:
- **xBD dataset** by Ritwik Gupta
  - [Paper](https://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.html)
  - [Dataset](https://xview2.org/dataset)
  - License: [CC BY-NC-SA 4.0]

- **BRIGHT dataset** by Hongruixuan Chen
  - [Repository](https://github.com/ChenHongruixuan/BRIGHT)
  - License: [CC BY-NC 4.0]


## License
All images and their associated annotations in DisasterM3 can be used for academic purposes only,
<font color="red"><b> but any commercial use is prohibited.</b></font>

<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">
<img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Junjue-Wang/DisasterM3&type=Date)](https://www.star-history.com/#Junjue-Wang/DisasterM3&Date)
