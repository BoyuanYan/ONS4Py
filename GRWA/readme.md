## GRWA

#### 1. 参数
在如下的参数设置中，ksp.sh得到的阻塞率在10%左右，一轮游戏的最终得分为5000分左右。
```bash
--net 6node.md --wave-num 10 --rou 8 --miu 300 --max-iter 300 \
--k 1 --weight None --workers 4 --steps 2700
```

|参数名|解释|默认值|
|:---|:---|:----|


#### 2. reward

见Service.py。

> 必须保证ARRIVAL_OP_OT的值和其他reward值不一样

|state|变量名（Service.py）|action|reward|
|:----|:-----|:-----|:-----|
|该时间点没有业务到达，可能有业务离去（取决于事件排序）| NOARRIVAL_NO|No-Action|1|
|该时间点没有业务到达，可能有业务离去（取决于事件排序）| NOARRIVAL_OT|选择其他RW选项|0|
|该时间点有业务到达（可能同时有业务离去），但是没有可达RW选项|ARRIVAL_NOOP_NO|No-Action|1|
|该时间点有业务到达（可能同时有业务离去），但是没有可达RW选项|ARRIVAL_NOOP_OT|选择其他RW选项|0|
|该时间点有业务到达（可能同时有业务离去），并且有可达RW选项|ARRIVAL_OP_OT|选择可达RW选项|10|
|该时间点有业务到达（可能同时有业务离去），并且有可达RW选项|ARRIVAL_OP_NO|选择不可达RW或者No-Action|-5|




####  3. 情况处理

1. 某时间点有一个业务到达和一条及以上业务离去，则根据事件排序结果，
先处理排在到达业务之前的离去业务，然后返回到达业务的observation
2. 某时间点只有业务离去，则直接进行业务离去处理，然后返回业务离去
之后的observation
3. 如果执行完某个行为以后，返回done=True，即本轮游戏结束。则自动reset()，重新开启一局
游戏，并且返回本次游戏的obs, reward，和done=True同时返回。由于done的作用仅仅是计算
return和reward，所以一次的误差不会带来什么影响。
4. 转成np.ndarray类型的图像CHW矩阵的时候，做了灰度图的归一化处理。
5. 使用graphviz生成图像然后用PIL.Image读入进行处理的方式，有以下两个弊端，目前并没有
比较可行的解决方案：
    1. 磁盘频繁读写造成效率低下，并且对磁盘的损伤比较大，很容易出现错误
    2. 不同帧图像覆盖的是同名文件，可能会出现图像文件还没有刷新好，就已经被读入的错误
> 作者已经回复了，详情可见：https://github.com/xflr6/graphviz/issues/49#issuecomment-370132248
> 已经在RwaNet.py中实现了。
6. 如果处理的到达业务是最后一个到达业务的话，则本游戏直接结束。因为后续只能是业务释放


#### 4. 优化想法

1. 把实时算出来的路由图像也加进去算作一个channel
2. 可以考虑把路由图像部分作为网络另一支路。
3. 现在是按照时间线一步一步前进，这样导致batch样本里面真正可用的只占`1.0/args.rou`，
是否应该修改成每一步都会直接推进到下一个业务到来的时刻？
> 已经实现了。增加了--step-over的选项，one_time表示按照时间序列推进，one_service表示
按照事件序列推进。
4. 增加模型验证的程序，其中注意act方法中的deterministic的取值，应该为True吧。

#### 5. 验证

1. 1channel_access_check 证明了GRWA具有判断网络可达性的能力
2. RL的action space，不同源宿下对应的路由是不一样的，这种隐性的逻辑尚未验证深度学习能否掌握


#### 6. 对比实验

1. 使用FC网络，对最小网络表达进行学习。需要对step函数进行修改
2. 增加ksp中计算出来的路由图像的channels，以此来明确给出路由信息。


---
---
#### 7. 踩过的坑

1. numpy中的布尔类型默认是numpy.bool_，这种类型不能使用“is True”这样的语句来判断真值。
要使用bool()做类型转换。
2. np.random.randint太坑了，使用Process多进程同时启动生成的随机数完全一样，搞得我还以为
是所有的进程共享一个游戏内存呢。改成了random模块的randint

#### 8. 基础模型

该工程的实现源自于[ikostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)
共享的利用pytorch实现A2C、PPO、ACKTR算法的工程，我fork到
[自己的仓库](https://github.com/BoyuanYan/pytorch-a2c-ppo-acktr)中，加入了一些测试结果
等等辅助理解的内容。

##### PongNoFrameskip-v4

模型有效性的判断是通过**PongNoFrameskip-v4**这个游戏进行评估的。
该游戏大体是玩乒乓球，在gym中没有找到名字对应的游戏，但是应该是 https://gym.openai.com/envs/Pong-v0/
 里面提到的那样。

| 参数名 | 参数取值 | 注释 |
|:------|:-------|:-----|
|action space| Discrete(6) | 每隔k帧进行一次action选择，k为{2,3,4}的均匀分布|
|observation space| Box(210, 160, 3)|HWC模式|
|reward range | (-inf, inf) ||

#### 9. 下一步要做的事情

1. 把该工程封装进gym里面去。
