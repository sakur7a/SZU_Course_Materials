# 建模资产查看器

查看 M0 资产包的 GLB、节点和交互。模型保留包内米制坐标，不会缩放或重设模型的变换；固定视角使用清单 `views`，缺少机位时只调整相机取景。

```powershell
npm ci
npm run dev -- --host 127.0.0.1
```

将完整资产包置于 `build/modeling-pipeline/web-public/assets/<asset-id>/`，通过 `http://127.0.0.1:5173/?asset=/assets/<asset-id>/manifest.json` 打开（实际端口以 Vite 输出为准）。清单中的 `model` 必须是资产目录内的相对路径。生产构建使用 `npm run build`，默认输出到 `build/modeling-pipeline/web-dist`；构建时 Vite 会复制 public 目录。`PIPELINE_PUBLIC_DIR` 和 `PIPELINE_WEB_OUT` 可分别覆盖 public 和输出目录，均为文件系统路径。输出目录不会自动清空。

使用 Khronos glTF Validator 验证自包含 GLB：

```powershell
node scripts/validate-glb.mjs path/to/model.glb path/to/report.json
```

不提供报告路径时 JSON 写入标准输出。有格式错误时退出码为 1，输入或运行错误时为 2。警告会保留在报告中；格式通过不代表视觉或语义验收通过。

浏览器自动化可使用 `window.__assetProbe`：`ready` 在清单、模型和必需节点成功加载后为真；`errors` 保存加载错误；`inspect()` 返回实际包围盒、锚点世界位置、节点和渲染统计；`setAngle(deg)` 和 `setView('hero'|'front'|'top')` 返回操作是否成功；`testHierarchy()` 暂时旋转清单指定父节点，比较后代世界矩阵并恢复，返回结果及前后矩阵。未加载时检查函数不会返回模拟结果。

实现参考：[Three.js GLTFLoader](https://threejs.org/docs/pages/GLTFLoader.html)、[OrbitControls](https://threejs.org/docs/pages/OrbitControls.html)、[Vite publicDir](https://vite.dev/config/shared-options.html#publicdir) 和 [Khronos glTF Validator](https://github.com/KhronosGroup/glTF-Validator)。
