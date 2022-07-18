* SCINet Added!
## 一些已知的issue
- [x] json里的部分键值对的key没有对齐，i.e. 有些地方是val，有些地方是valid
- [x] json的语法和python不一样，true, false都是小写开头，None应该变成null
- [x] SCINet的dataloader里面的部分依赖在util里面还没有实现
- [ ] 关于device，有些地方的缺少.to(device)，目前只能在cpu上运行
- [ ] SCINet的dataloader和REPO里面的还没有对齐，目前暂时忽略了label_len但是仍待解决

## TODO
- [ ] 添加一个json之间的互相引用机制，模仿mmcv系列里面的__base__
- [ ] dataloader支持timestamp


## 本周任务：把SCINet加入到REPO中，并实现ETTh1
### 参考步骤（不一定全）
1. 把SCINet model文件复制粘贴到/models/SCINet/SCINet.py下
2. 往cfgs/model里加一个scinet.json，里面陈列scinet使用到的所有参数，默认值可以按ETTh1，多变量horizon=24的
3. 往cfgs/data里加一个ETTh1.json，里面放ETTh1的基础设置
4. 往cfgs/exp/scinet里加一个scinet_etth1_multivariate_h24.json, 参考cfgs/exp/basic.json，模型参数换成scient的，数据集参数换成etth1的
5. run.py里读取的cfg改为scinet_etth1_multivariate_h24.json
6. data_processing/data_handler.py里加入ETTh的dataset，可以直接参考scinet里使用的，参数记得和config里匹配上
7. 调整models/SCINet/SCINet.py与cfg对应，应该可以运行了
