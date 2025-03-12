# FengWu-W2S.onnx 
FengWu-W2S:Revisiting the potential for seamless forecasting in AI Weather Models

## Demos
### Demo 1: hourly_6-hour precipitation (tp6h)
![tp6h](./demos/hourly_tp6h.gif)

### Demo 2: weekly_6-hour precipitation (tp6h)
![tp6h_week](./demos/weekly_tp6h.gif)


### Demo 3: 25km Typhoon 850hPa Wind Speed initial from 2024-06-21T00 
![Typhoon](./demos/EA_Typhoon.gif)
*You can observe the generation process of multiple typhoons in late July, as highlighted in member 23.*


## Onnx link

We offer two versions of our model:

1. **140 km Resolution**: For access to this version, please contact us at [lingfenghua@pjlab.org.cn](mailto:lingfenghua@pjlab.org.cn) and [bailei@pjlab.org.cn](mailto:bailei@pjlab.org.cn).
   
2. **25 km Resolution**: This version includes stratospheric data and provides enhanced spatial detail, making it particularly effective for predicting extreme weather events, similar to the results shown in Demo 3. The 25 km model will be open-sourced following the acceptance of the manuscript.

Feel free to reach out if you would like to obtain either version.

## Citation
```
@article{Ling2024,
  title={FengWu-W2S: A deep learning model for seamless weather-to-subseasonal forecast of global atmosphere},
  author={Fenghua Ling and Kang Chen and Jiye Wu and Tao Han and Jing-Jia Luo and Wanli Ouyang and Lei Bai},
  journal={arXiv preprint arXiv:2411.10191},
  year={2024},
  url={https://arxiv.org/abs/2411.10191}
  }
