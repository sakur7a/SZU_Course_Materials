# 计算机图形学期末大作业：深圳大学粤海校区虚拟场景

基于 C++11、OpenGL、GLFW 和 GLM 的校园场景课程项目。场景以开放地图和照片资料为参考，建筑外观、地形及高程有近似，不是实测校园模型。项目仍在完善中。

## 目录

- `main/end/code/`：场景程序、网格和纹理资源、生成脚本与独立 CMake 构建配置。
- `main/end/modeling/`：建模资产、验收记录、查看器及工作交接。
- `main/end/figures/` 和 `main/end/预览资料/`：阶段截图、来源及旧预览使用说明。预览资料不含可执行文件。
- `main/end/PIPELINE.md`、课程要求及答题纸/资源说明模板：建模流程和课程材料。
- `tools/end/`：场景资产和建模流程脚本；未包含本机生成缓存。

目录保留原工作区的 `main/end` 与 `tools/end` 相对关系。这里仅包含期末大作业，原工作区的其他实验和统一 CMake 配置未包含。

## 构建与运行

需要支持 C++11 的编译器、CMake、GLFW、GLAD 和 GLM。以下是 Windows、Ninja 与 vcpkg 静态 MinGW 配置的示例；需先安装对应工具、将匹配的编译器加入 PATH，并安装 vcpkg 的 `glad`、`glfw3`、`glm` 包：

```powershell
# 在本目录执行；VCPKG_ROOT 指向本机的 vcpkg 目录
$toolchain = Join-Path $env:VCPKG_ROOT 'scripts/buildsystems/vcpkg.cmake'
cmake -S main/end/code -B build/campus -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_TOOLCHAIN_FILE=$toolchain" -DVCPKG_TARGET_TRIPLET=x64-mingw-static
cmake --build build/campus
.\build\campus\szu-campus.exe
```

构建配置会将 `assets` 和 `shaders` 复制到可执行文件旁。程序自检可执行 `.\build\campus\szu-campus.exe --self-test`。其他编译器与平台需要配置相应 vcpkg triplet 和工具链。生成全部场景资源还需要脚本列出的 Python 图形和地理数据依赖；使用现有资源运行程序不需要 Python。

## 当前状态

- 元平体育馆入口和体育场西北入口已做局部建模及路线检查；其他建筑的入口、道路接缝和可行走面仍需逐处核验。
- 这次协作快照包含北馆入口和二层的候选修改：源码以及 14 个新增和 2 个更新的资源来自本机最近的校园集成构建。候选路线检查通过，但尚未完成最终视觉与交互验收，也尚未同步到本机旧预览版。
- `modeling/CURRENT_WORK.md` 等历史记录形成于上述候选阶段之前；北馆的最新状态以上一条为准。
- 贴地行走是相机高度辅助，尚非完整碰撞物理；材料与尺寸应以来源记录和实地情况继续核实。
- 2026-11-16 在 Windows + MinGW + vcpkg 环境从本快照独立构建并完成一次离屏渲染；程序自检、北馆候选及元平、体育场入口的专项路线检查通过。这些检查不代表完整人工验收。
- 课程过程报告、正式使用说明书和最终提交包尚待完成；本仓库是协作快照，不是最终提交包。

## 操作与资料

按住鼠标左键拖动调整视线；`W/A/S/D` 移动，`Q/E` 升降，`Shift+T` 切换贴地行走。方向键控制广场装置，`J/L` 控制小巴。更多预设视角与交互见 `main/end/预览资料/使用说明.txt`；其中描述的是较早的预览版，北馆候选变动不在其中。

地图及模型来源见 `main/end/code/data/SOURCES.md`、`main/end/code/assets/CREDITS.txt` 与各建筑资料。使用外部资源时请继续保留其许可和署名。
