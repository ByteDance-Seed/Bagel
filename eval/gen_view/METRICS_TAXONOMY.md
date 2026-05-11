# Novel View Generation 评测指标体系（重构版）

## 1. 细粒度几何-位姿层（Pose & Geometry）
目标：评估目标视角是否“拍对位置 + 拍对方向”。

1. `pose_t_err_m`
定义：预测目标相机中心与 GT 目标相机中心的欧氏距离（米）。

2. `pose_r_err_deg`
定义：预测目标相机旋转与 GT 旋转的测地角误差（度）。

3. `rpe_trans_err_m`
定义：以最后一帧 context 为参考的相对平移误差。

4. `rpe_trans_dir_cos`
定义：预测与 GT 相对平移方向的余弦相似度，越接近 1 越好。

5. `rpe_rot_deg`
定义：以最后一帧 context 为参考的相对旋转误差（度）。

6. `ctx_align_rmse_m`
定义：用 Umeyama 对齐后，预测 context 轨迹与 GT context 轨迹的 RMSE。

7. `ctx_scale`
定义：预测轨迹到 GT 轨迹的最优缩放因子。

## 2. 细粒度像素层（Photometric Fidelity）
目标：评估渲染结果与 GT 图片在像素域的一致性。

1. `pix_mae`
定义：逐像素绝对误差均值。

2. `pix_rmse`
定义：逐像素均方根误差。

3. `pix_psnr`
定义：峰值信噪比，越高越好。

4. `pix_ssim`
定义：结构相似性（若环境缺少 `skimage` 则为 NaN）。

## 3. 细粒度三维重建层（Point Cloud Consistency）
目标：评估由图像恢复出的几何结构是否与 GT 视角恢复结果一致。

1. `pc_chamfer`
定义：预测点云与 GT 点云的双向 Chamfer 距离（越小越好）。

2. `pc_fscore`
定义：在阈值 `POINT_F_THRESH` 下的双向 F-score（越大越好）。

说明：当前实现默认使用 VGGT 的 `point_head`，若不可用则尝试 `depth_head` 回退；两者都不可用时该组指标为 NaN。

## 4. 粗粒度场景语义-关系层（Object & Spatial Relation）
目标：评估生成视角下“看到哪些物体”和“物体关系”是否与 GT 一致。

1. `obj_precision / obj_recall / obj_f1 / obj_iou`
定义：基于 GT 可见物体集合与预测视角投影可见集合计算的集合指标。

2. `obj_rel_acc`
定义：共享物体对之间 left-right / up-down / near-far 关系一致率。

3. `obj_orient_err_deg`
定义：共享物体相对相机朝向（相对航向角）误差均值（度）。

实现说明：
1. 若数据含 3D 物体标注（`instances` + `visible_instance_ids`），优先走 3D 几何投影评测。
2. 若场景无物体标注或强制 `--object_source detector`，自动走开放词表检测（OWLv2）评测：
   - 指标仍输出 `obj_precision/recall/f1/iou/rel_acc`。
   - `obj_orient_err_deg` 在纯检测路径下默认为 NaN（2D 检测无法稳定恢复真实 3D 朝向）。

## 5. 关键改进：引入视频帧 pose 先验
当前代码在预测目标 pose 时，不再只依赖“裸 VGGT 输出”，而是融合 context 的 GT pose 先验：

1. 用 Umeyama 将预测 context 轨迹对齐到 GT context 轨迹。
2. 用“最后一帧 context 的 GT pose”锚定预测相对变换（relative pose anchoring）。
3. 将上述两路结果做平移融合（`POSE_PRIOR_BLEND` 可调）。

效果：可显著减少全局漂移和尺度漂移，提高目标帧 pose、点云和对象关系指标稳定性。

## 6. 高精度后端策略（Accuracy First）
新增 `POSE_BACKEND`：
1. `auto`：优先 COLMAP（若环境存在 `colmap`），失败自动回退 VGGT。
2. `colmap`：强制 COLMAP SfM（更高精度但更慢）。
3. `vggt`：强制 VGGT（更快，精度通常低于 SfM）。

COLMAP 路径下：
1. 使用 context + target 图像做 SfM，提取目标相机 pose。
2. 支持从重建稀疏点中提取 target 可见点用于 `pc_chamfer/pc_fscore`。
3. 若 COLMAP 不可用或重建失败，`auto` 自动回退，不中断全流程。

## 7. 输出文件
1. `metrics_final.csv`
内容：逐样本全量指标。

2. `metrics_summary.json`
内容：按四层指标分组聚合后的均值摘要。
