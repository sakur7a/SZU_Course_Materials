# 贡献指南

感谢你对本项目的关注！这个仓库离不开大家的贡献。如果你有期末试卷、实验报告、课程笔记或其他有用的资料，欢迎分享给学弟学妹们。

## 可以贡献什么？

任何形式的贡献都非常欢迎，包括但不限于：

- :material-file-document: 补充现有课程的资料（试卷、作业、笔记等）
- :material-folder-plus: 添加新的课程目录
- :material-bug: 修正已有资料中的错误
- :material-text-box-edit: 完善课程说明和描述

## 方式一：提交 Pull Request（推荐）

1. **Fork** 本仓库到你的 GitHub 账号
2. **Clone** 你 Fork 的仓库到本地
   ```bash
   git clone https://github.com/<你的用户名>/SZU_Course_Materials.git
   ```
3. 创建新分支并添加你的资料
   ```bash
   git checkout -b add-my-materials
   ```
4. 将文件放入对应的课程目录下，如果是新课程请按现有结构创建目录：
    - **专业课** → `专业课/<课程名>/`
    - **通识课** → `通识课/<课程名>/`
5. 提交并推送
   ```bash
   git add .
   git commit -m "feat: 添加 XX 课程资料"
   git push origin add-my-materials
   ```
6. 在 GitHub 上发起 **Pull Request**，简要描述你添加的内容

!!! tip "提示"
    推送到 `main` 分支后，GitHub Actions 会**自动**运行构建脚本并更新网站，无需手动部署。

## 方式二：提交 Issue

如果你不熟悉 Git 操作，可以直接在 [Issues](https://github.com/sakur7a/SZU_Course_Materials/issues) 中上传文件或提供资料链接，我会帮你整理并添加。

## 目录结构说明

```
SZU_Course_Materials/
├── 专业课/
│   └── <课程名>/
│       ├── readme.md        # 课程说明（可选）
│       ├── 往年试卷/
│       ├── 实验报告/
│       └── ...
├── 通识课/
│   └── <课程名>/
│       └── ...
├── 转专业/
└── 模版与表格/
```

## 注意事项

!!! warning "版权声明"
    - 请确保分享的资料**不会侵犯他人版权**
    - 建议使用有意义的文件名，避免 `新建文档.docx` 之类的命名
    - 教师课件请在获得许可后上传
