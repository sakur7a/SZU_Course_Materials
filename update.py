#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import hashlib
import shutil
from urllib.parse import quote

# 仓库信息
REPO = 'sakur7a/SZU_Course_Materials'
TXT_URL_PREFIX = f'https://github.com/{REPO}/blob/main/'
BIN_URL_PREFIX = f'https://github.com/{REPO}/raw/main/'

# 需要扫描的顶级目录（分类目录）
CATEGORY_DIRS = ['专业课', '通识课', '模版与表格', '转专业']
EXCLUDE_DIRS = ['.git', 'docs', '.vscode', 'site', '.github', '__pycache__']
README_MD = ['README.md', 'readme.md', 'index.md', 'READMD.md']
TXT_EXTS = ['md', 'txt', 'py', 'cpp', 'c', 'h', 'java', 'asm', 'js', 'ts', 'm']

# 临时隐藏：这些课程目录不参与文档与导航生成
EXCLUDE_COURSE_ENTRIES = {
    '专业课/机器学习',
    '专业课/计算机网络',
}

# 用于生成稳定的 ASCII 文件名
_slug_counter = 0
_slug_map = {}

# 为常见中文目录提供可读 URL（可按需继续补充）
SLUG_OVERRIDES = {
    '专业课': 'major-courses',
    '通识课': 'general-courses',
    '模版与表格': 'templates-and-forms',
    '转专业': 'major-transfer',
    '专业课/专业英语': 'professional-english',
    '专业课/人工智能导论': 'intro-to-ai',
    '专业课/数据结构': 'data-structures',
    '专业课/最优化方法': 'optimization-methods',
    '专业课/计算机系统（1）': 'computer-systems-1',
    '专业课/计算机系统（2）': 'computer-systems-2',
    '专业课/计算机视觉': 'computer-vision',
    '专业课/数字电路': 'digital-circuits',
    '专业课/算法设计与分析': 'algorithm-design-and-analysis',
    '通识课/大学物理': 'college-physics',
    '通识课/大学物理实验（1）': 'college-physics-lab-1',
    '通识课/概率论与数理统计': 'probability-and-statistics',
    '通识课/线性代数': 'linear-algebra',
    '通识课/高等数学': 'advanced-mathematics',
    '转专业/往年原题': 'transfer-past-exams',
    '转专业/经验': 'transfer-experience',
}


def _ascii_slug(text):
    """将任意文本转换为小写 ASCII slug（尽量可读）。"""
    text = text.lower()
    # 常见连接符统一为短横线
    text = re.sub(r'[\s_/\\（）()\[\]{}]+', '-', text)
    # 仅保留英文、数字和短横线
    text = re.sub(r'[^a-z0-9-]+', '', text)
    text = re.sub(r'-{2,}', '-', text).strip('-')
    return text

def to_slug(name):
    """将目录名转为稳定且可读的 ASCII slug。"""
    global _slug_counter
    if name in _slug_map:
        return _slug_map[name]

    if name in SLUG_OVERRIDES:
        base_slug = SLUG_OVERRIDES[name]
    else:
        base_slug = _ascii_slug(name)

    if not base_slug:
        # 如果无法从原名提取 ASCII，可回退到稳定 hash 前缀
        h = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
        base_slug = 'page-{}'.format(h)

    slug = base_slug
    dedupe = 2
    while slug in _slug_map.values():
        slug = '{}-{}'.format(base_slug, dedupe)
        dedupe += 1

    _slug_map[name] = slug
    return slug


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

    # 复制 pages/ 目录下的静态页面到 docs/
    if os.path.isdir('pages'):
        for f in os.listdir('pages'):
            if f.endswith('.md'):
                shutil.copy2(os.path.join('pages', f), os.path.join('docs', f))

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
            if '{}/{}'.format(category, entry) in EXCLUDE_COURSE_ENTRIES:
                continue
            if not os.listdir(entry_path):
                continue

            # 生成 ASCII 文件名避免 GitHub Pages URL 编码问题
            slug = to_slug(category + '/' + entry)
            md_filename = '{}.md'.format(slug)
            output_path = os.path.join('docs', md_filename)
            generate_md(entry, entry_path, output_path)
            category_items.append({entry: md_filename})
            category_courses[category].append((entry, md_filename))

        # 如果该分类下没有子目录，检查是否有直接文件
        if not category_items:
            # 将整个分类目录当作一个条目
            slug = to_slug(category)
            md_filename = '{}.md'.format(slug)
            output_path = os.path.join('docs', md_filename)
            generate_md(category, category, output_path)
            nav_items.append({category: md_filename})
            category_courses[category].append((category, md_filename))
        else:
            nav_items.append({category: category_items})

    # 添加额外页面到 nav
    nav_items.append({'贡献指南': 'contributing.md'})
    nav_items.append({'友情链接': 'links.md'})
    nav_items.append({'更新日志': 'changelog.md'})

    # 生成首页 docs/index.md
    generate_index(category_courses)

    # 更新 mkdocs.yml 的 nav 部分
    write_mkdocs_nav(nav_items)

    print('✅ docs/ 目录已生成')
    print('nav 结构:')
    for item in nav_items:
        print(' ', item)


def parse_readme_tables():
    """从 README.md 解析课程表格，提取时间和内容信息"""
    with open('README.md', 'r', encoding='utf-8') as f:
        readme_content = f.read()

    # 提取课程名 -> {时间, 内容} 的映射
    course_info = {}
    # 匹配表格行：| [课程名](链接) | 时间 | 内容 | 或 | [内容](链接) | 说明 |
    for line in readme_content.split('\n'):
        line = line.strip()
        if not line.startswith('|') or line.startswith('| ---') or line.startswith('| 课程') or line.startswith('| 内容'):
            continue
        cells = [c.strip() for c in line.split('|')]
        # cells[0] is empty (before first |), cells[-1] is empty (after last |)
        cells = [c for c in cells if c != '']
        if len(cells) < 2:
            continue
        # 从第一列提取课程名（可能是 [名称](链接) 格式）
        name_cell = cells[0]
        match = re.search(r'\[([^\]]+)\]', name_cell)
        name = match.group(1) if match else name_cell
        if len(cells) == 3:
            course_info[name] = {'时间': cells[1], '内容': cells[2]}
        elif len(cells) == 2:
            course_info[name] = {'时间': '', '内容': cells[1]}

    return course_info


def generate_index(category_courses):
    """生成 MkDocs 首页，包含课程目录和原 README 内容"""
    lines = []

    # 从 README 解析课程信息
    course_info = parse_readme_tables()

    # 提取标题和前言
    lines.append('# 深圳大学 CS 本科课程资料共享\n\n')
    lines.append('[![GitHub stars](https://img.shields.io/github/stars/{repo}?style=social)](https://github.com/{repo})\n'.format(repo=REPO))
    lines.append('[![GitHub forks](https://img.shields.io/github/forks/{repo}?style=social)](https://github.com/{repo}/fork)\n'.format(repo=REPO))
    lines.append('[![GitHub last commit](https://img.shields.io/github/last-commit/{repo})](https://github.com/{repo}/commits/main)\n\n'.format(repo=REPO))
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

    # 专业课和通识课显示时间和内容列
    categories_with_time = {'专业课', '通识课'}

    for category in CATEGORY_DIRS:
        if category not in category_courses:
            continue
        courses = category_courses[category]
        label = category_labels.get(category, '### ' + category)
        lines.append(label + '\n\n')

        if category in categories_with_time:
            lines.append('| 课程 | 时间 | 内容 | 链接 |\n')
            lines.append('| --- | --- | --- | --- |\n')
            for name, md_filename in courses:
                info = course_info.get(name, {})
                time_str = info.get('时间', '')
                content_str = info.get('内容', '')
                lines.append('| {} | {} | {} | [查看详情]({}) |\n'.format(
                    name, time_str, content_str, md_filename))
        else:
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
