# 西侧校门模型依据

立德门、立业门与立志门均独立输出模型文件，生成逻辑为 `campus_gates_west.py`。
资料中可辨的构造与模型估计分开记录；照片只用于观察，未作为发布纹理。

## 立德门

- 学校外国语学院校园照片：<https://sfl.szu.edu.cn/xysh/xyfg.htm>，直接图片 <https://sfl.szu.edu.cn/images/XYFG/1-201112091426.jpg>。当前工作区参考图 `build/campus-research-reference/sfl-gate.jpg` 显示不同形态的双高墙、北侧开口、密集横向分缝、挑檐、竖排校名与蓝色门牌。
- 2022 年校方航拍转载：<https://www.thepaper.cn/newsDetail_forward_18845502>；同图 <https://sz.oeeee.com/html/202207/03/1253587.html>。可见门柱和两条车行线、中央小岛及长草坪，临时帐篷未作为永久模型。
- 校方 2024 年车辆门禁系统升级通知：<https://baowei.szu.edu.cn/gzdt/5.htm>。因此照片中的门禁形式不能视为现状保证。
- 沿用现有模型中心 `(-410,-54)`，门柱及岗亭按各自地面接地。模型提高横缝、玻璃分格、门窗框、屋檐底面、设备尺度等近景细节；高度、具体窗格、门禁数量和构造尺寸仍为比例估计。校名用华文行楷近似，非原作书法。

## 立业门（小西门）

- 学校 2024 新生指南明确“立业门（小西门）”：<https://zs.szu.edu.cn/__local/D/5D/DB/DE1FA1FE903EA17D2A8BBECF2C1_AEAB0EA1_FA1E90.pdf>。
- 校方 2025 年交通通知再次将其列为出入校门：<https://baowei.szu.edu.cn/info/1003/1381.htm>；通知仅用于身份和路网关系，临时通行管制未写入场景。
- 2021 年学生路线图：<https://m.sohu.com/a/488635413_121124309>。定位图和手绘图指向南海大道、老西南餐厅以西的入口。下载的参考图片只供研究，未打包。
- OSM `node/2398019533`：<https://www.openstreetmap.org/node/2398019533>；岗亭 `way/241588751`：<https://www.openstreetmap.org/way/241588751>。模型采用其入口位置及小型矩形岗亭 footprint，拆分墙体、玻璃、屋檐、窗框和设备；根生成器应去除该岗亭通用盒体以免重复。
- 2026 年学校新闻 <https://news.szu.edu.cn/info/1003/15183.htm> 明确立业门紧邻公交站并有保卫室，页面含门口监控 GIF。已从公开图片镜像 <https://www.szu.edu.cn/__local/8/55/00/419750882BF8ADE2DB794B79F31_1B12CAC8_5841D0.gif> 和 <https://www.szu.edu.cn/__local/C/90/3F/EB87C6781371505AA8889BF3C06_A267FAC0_6EBBB5.gif> 获取首帧供观察，可辨酒红色门禁设备、黄黑防撞短柱、收拢的金属伸缩门和黄黑减速带。模型据此增加这些细节，数量、位置及尺寸仍为道路尺度估计。画面白色“深圳大学粤海”是监控水印，没有作为建筑文字复制。
- ArcGIS World Imagery <https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer> 仅供观察，树冠和影像年代限制了入口构件的辨认。
- **尚未确认**：完整门头、固定栏杆边界、人行闸机数量、岗亭窗框和门禁设备的精确点位。当前细节与小型门名牌为可替换近似，未添加没有证据的大门楼。

## 立志门

- 用户提供的 2023 校园地图明确标有立志门，位于深南大道一侧、文科教学区以北。
- 地图 POI：<https://ditu.amap.com/place/B0GDDMS9L6>。高德坐标为其自身坐标系，不直接当作本项目 WGS84 数据使用；仅与手绘图和校园北侧边界凹口对照位置。
- OSM 近旁连接线 `way/1271852288`：<https://www.openstreetmap.org/way/1271852288>，快照仍为 `highway=construction`，不足以证明当前通行状态和门楼外貌。
- 未找到可独立确认完整永久立面的地面照片，因此**仅生成写明“位置参考”的小标记牌**，没有杜撰完整门头或新道路。该模型是待细化的位置标记，不计作精确复原校门。

## 资源及接入

- 每门按 `stone`、`metal`、`glass`、`roof`、`sign` 分层（不需要的层省略），均保存为独立 `campus-gate-<id>-<material>.mesh`；立业门另有红色、黄色金属层，`layers` 中仍使用 `material=metal` 并按层给出颜色和独立文件名，调用方不能按 material 去重。所有文字与门牌均由 Pillow 重绘。
- 应停止绘制旧 C++ `lideGate(...)` 调用，保留其他校门函数由对应负责人处理。
- 新模型不修改道路、商铺和全校地形。`vegetation_keep_clear` 返回主要实体 footprint，根生成器可用于随机植被避让。
