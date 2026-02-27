#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from urllib.parse import quote

# 仓库信息
REPO = 'sakur7a/SZU_Course_Materials'
TXT_URL_PREFIX = f'https://github.com/{REPO}/blob/main/'
BIN_URL_PREFIX = f'https://github.com/{REPO}/raw/main/'

# 需要扫描的顶级目录（分类目录）
CATEGORY_DIRS = ['专业课', '通识课', '模版与表格', '转专业']
EXCLUDE_DIRS = ['.git', 'docs', '.vscode', 'site', '.github', '__pycache__']
README_MD = ['README.md', 'readme.md', 'index.md']
TXT_EXTS = ['md', 'txt', 'py', 'cpp', 'c', 'h', 'java', 'asm', 'js', 'ts', 'm']


def make_link(root, filename):
    """生成指向 GitHub 的文件链接"""
    rel_path = '{}/{}'.format(root, filename).replace('\\', '/')
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext in TXT_EXTS:
        return '[{}]({})'.format(filename, TXT_URL_PREFIX + quote(rel_path))
    else:
        return '[{}]({})'.format(filename, BIN_URL_PREFIX + quote(rel_path))


def list_files(course_path):
    """遍历课程目录，生成文件列表的 Markdown 文本"""
    filelist_texts = '## 文件列表\n\n'
    readme_path = ''

    for root, dirs, files in os.walk(course_path):
        # 排除隐藏目录和特殊目录
        dirs[:] = [d for d in sorted(dirs) if d not in EXCLUDE_DIRS and not d.startswith('.')]
        files.sort()

        level = root.replace(course_path, '').count(os.sep)
        indent = ' ' * 4 * level
        folder_name = os.path.basename(root)
        filelist_texts += '{}- **{}**\n'.format(indent, folder_name)
        subindent = ' ' * 4 * (level + 1)

        for f in files:
            if f in README_MD:
                if root == course_path and readme_path == '':
                    readme_path = os.path.join(root, f)
            else:
                filelist_texts += '{}- {}\n'.format(subindent, make_link(root, f))

    return filelist_texts, readme_path


def generate_md(course_name, course_path, output_path):
    """为每门课程生成对应的 docs/*.md 文件"""
    filelist_texts, readme_path = list_files(course_path)
    final_texts = []

    if readme_path:
        with open(readme_path, 'r', encoding='utf-8') as file:
            final_texts = file.readlines()

    final_texts.append('\n\n')
    final_texts.append(filelist_texts)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.writelines(final_texts)


def main():
    if not os.path.isdir('docs'):
        os.mkdir('docs')

    # 收集 nav 结构
    nav_items = []
    nav_items.append({'首页': 'index.md'})

    # 用于生成首页课程目录的数据
    category_courses = {}

    for category in CATEGORY_DIRS:
        if not os.path.isdir(category):
            continue

        category_items = []
        category_courses[category] = []
        sub_entries = sorted(os.listdir(category))

        for entry in sub_entries:
            entry_path = os.path.join(category, entry)
            if not os.path.isdir(entry_path):
                continue
            if entry in EXCLUDE_DIRS:
                continue

            # 生成 md 文件名：使用 "分类_课程名" 避免冲突
            md_filename = '{}_{}.md'.format(category, entry)
            output_path = os.path.join('docs', md_filename)
            generate_md(entry, entry_path, output_path)
            category_items.append({entry: md_filename})
            category_courses[category].append((entry, md_filename))

        # 如果该分类下没有子目录，检查是否有直接文件
        if not category_items:
            # 将整个分类目录当作一个条目
            md_filename = '{}.md'.format(category)
            output_path = os.path.join('docs', md_filename)
            generate_md(category, category, output_path)
            nav_items.append({category: md_filename})
            category_courses[category].append((category, md_filename))
        else:
            nav_items.append({category: category_items})

    # 生成首页 docs/index.md
    generate_index(category_courses)

    # 更新 mkdocs.yml 的 nav 部分
    write_mkdocs_nav(nav_items)

    print('✅ docs/ 目录已生成')
    print('nav 结构:')
    for item in nav_items:
        print(' ', item)


def generate_index(category_courses):
    """生成 MkDocs 首页，包含课程目录和原 README 内容"""
    lines = []

    # 读取 README.md 的前言部分
    with open('README.md', 'r', encoding='utf-8') as f:
        readme_content = f.read()

    # 提取标题和前言
    lines.append('# 深圳大学 CS 本科课程资料共享\n\n')
    lines.append('!!! note "关于"\n')
    lines.append('    初衷是因为腾班的一些课程比较封闭，前人的经验也很少，希望这个 repo 可以帮到大家。\n')
    lines.append('    我一般期末考完后会 update，然后可能有一些经验...\n')
    lines.append('    更多资料可参考：[SZU_Math_and_Computer](https://github.com/Hytidel/SZU_Math_and_Computer)\n\n')

    # 课程目录
    lines.append('## 课程目录\n\n')

    category_labels = {
        '专业课': '### 💻 专业课',
        '通识课': '### 📐 通识课 / 基础课',
        '模版与表格': '### 📋 模版与表格',
        '转专业': '### 🔄 转专业',
    }

    for category in CATEGORY_DIRS:
        if category not in category_courses:
            continue
        courses = category_courses[category]
        label = category_labels.get(category, '### ' + category)
        lines.append(label + '\n\n')

        lines.append('| 课程 | 链接 |\n')
        lines.append('| --- | --- |\n')
        for name, md_filename in courses:
            lines.append('| {} | [查看详情]({}) |\n'.format(name, md_filename))
        lines.append('\n')

    # 许可
    lines.append('## 许可\n\n')
    lines.append('大部分是我的实验报告和作业，可能会有些往年的考题或者资料。\n\n')
    lines.append('有些资料来自网络，如有可能的侵权行为麻烦您联系 sakur7a@outlook.com，带来的不便请您谅解。\n\n')
    lines.append('资料仅供参考，请自行判断其适用性。\n')

    with open('docs/index.md', 'w', encoding='utf-8') as f:
        f.writelines(lines)


def write_mkdocs_nav(nav_items):
    """将 nav 写入 mkdocs.yml"""
    import yaml

    with open('mkdocs.yml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['nav'] = nav_items

    with open('mkdocs.yml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    main()
