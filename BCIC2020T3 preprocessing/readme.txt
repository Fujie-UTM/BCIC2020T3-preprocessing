1. 想象动作起始为t=0, 无基线修正

2. epoch包含[-0.5s 2.6s]：
[-0.5s 0s]为准备阶段，屏幕显示“+”
[0s, 2s]为想象阶段，屏幕为空白
[2s, 2.6s]为下一个trial的准备阶段，屏幕显示“+”

3. 工频噪声为60Hz

4. epoch采样率Fs=256 Hz，LPF=128 Hz

5. 被试数：15人

6. 试次数量：train：60 trials/class，validation：10 trials/class

7. 任务：5分类，"Hello":0,"Help me":1,"Stop":2,"Thank you":3,"Yes":4

8. montage：
epoch：共参考，standard_1020
原始：BrainVision 64导系统，Fpz=GND，FCz=Ref，

