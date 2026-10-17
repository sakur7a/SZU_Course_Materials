import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..", "..", "..", "..");
const OUT_DIR = path.join(ROOT, "outputs", "closest-pair-oral");
const SCRATCH_DIR = path.join(ROOT, "tmp", "slides", "closest-pair-oral");
const PREVIEW_DIR = path.join(SCRATCH_DIR, "preview");
const INSPECT_PATH = path.join(SCRATCH_DIR, "inspect.ndjson");

const W = 1280;
const H = 720;
const TOTAL = 7;

const C = {
  bg: "#FFFFFF",
  ink: "#0F172A",
  text: "#111827",
  sub: "#5B6472",
  line: "#D9E1EA",
  line2: "#EAEFF4",
  blue: "#2563EB",
  blueSoft: "#EEF4FF",
  green: "#0F9D6A",
  greenSoft: "#F1FDF8",
  red: "#DC2626",
  graySoft: "#F8FAFC",
};

const FONT = {
  title: "Poppins",
  body: "Lato",
  mono: "Aptos Mono",
};

const inspectRecords = [];

async function ensureDirs() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addShape(slide, geometry, position, fill = C.bg, line = { fill: C.line, width: 1 }) {
  return slide.shapes.add({ geometry, position, fill, line });
}

function addText(
  slide,
  slideNo,
  text,
  left,
  top,
  width,
  height,
  {
    fontSize = 20,
    color = C.text,
    bold = false,
    typeface = FONT.body,
    align = "left",
    valign = "top",
    fill = "#00000000",
    line = { fill: "#00000000", width: 0 },
    role = "text",
  } = {},
) {
  const shape = addShape(slide, "rect", { left, top, width, height }, fill, line);
  shape.text = text;
  shape.text.fontSize = fontSize;
  shape.text.color = color;
  shape.text.bold = bold;
  shape.text.typeface = typeface;
  shape.text.alignment = align;
  shape.text.verticalAlignment = valign;
  shape.text.insets = { left: 0, right: 0, top: 0, bottom: 0 };
  inspectRecords.push({
    kind: "textbox",
    slide: slideNo,
    role,
    text: String(text),
    textChars: String(text).length,
    textLines: String(text).split(/\n/).length,
    bbox: [left, top, width, height],
  });
  return shape;
}

async function addImage(slide, slideNo, imagePath, position, role, fit = "contain") {
  const image = slide.images.add({
    blob: await readImageBlob(imagePath),
    fit,
    alt: role,
  });
  image.position = position;
  inspectRecords.push({
    kind: "image",
    slide: slideNo,
    role,
    path: imagePath,
    bbox: [position.left, position.top, position.width, position.height],
  });
  return image;
}

function addTopSystem(slide, slideNo, section, page) {
  addText(slide, slideNo, section.toUpperCase(), 72, 30, 320, 20, {
    fontSize: 12,
    color: C.blue,
    bold: true,
    typeface: FONT.mono,
    role: "system-section",
  });
  addText(slide, slideNo, `${String(page).padStart(2, "0")}`, 1152, 24, 56, 34, {
    fontSize: 24,
    color: C.ink,
    bold: true,
    typeface: FONT.title,
    align: "right",
    role: "system-page",
  });
  addText(slide, slideNo, `/ ${String(TOTAL).padStart(2, "0")}`, 1208, 33, 40, 18, {
    fontSize: 12,
    color: C.sub,
    bold: true,
    typeface: FONT.mono,
    align: "left",
    role: "system-page-total",
  });
  addShape(slide, "rect", { left: 72, top: 64, width: 1136, height: 1 }, C.line2, { fill: "#00000000", width: 0 });
}

function addTitleBlock(slide, slideNo, title, subtitle, x = 72, y = 92, w = 920) {
  addText(slide, slideNo, title, x, y, w, 66, {
    fontSize: 30,
    color: C.ink,
    bold: true,
    typeface: FONT.title,
    role: "title",
  });
  if (subtitle) {
    addText(slide, slideNo, subtitle, x, y + 78, w, 36, {
      fontSize: 16,
      color: C.sub,
      typeface: FONT.body,
      role: "subtitle",
    });
  }
}

function addStatement(slide, slideNo, text, x, y, w, accent = C.blue) {
  addShape(slide, "rect", { left: x, top: y, width: 6, height: 54 }, accent, { fill: "#00000000", width: 0 });
  addText(slide, slideNo, text, x + 18, y + 2, w - 18, 56, {
    fontSize: 22,
    color: C.ink,
    bold: true,
    typeface: FONT.title,
    role: "statement",
  });
}

function addBulletList(slide, slideNo, items, left, top, width, lineHeight = 42, fontSize = 18, role = "bullets") {
  items.forEach((item, idx) => {
    addShape(slide, "ellipse", { left, top: top + idx * lineHeight + 9, width: 8, height: 8 }, C.blue, { fill: "#00000000", width: 0 });
    addText(slide, slideNo, item, left + 20, top + idx * lineHeight, width - 20, lineHeight, {
      fontSize,
      color: C.text,
      typeface: FONT.body,
      role,
    });
  });
}

function addMiniCard(slide, slideNo, x, y, w, h, value, label, accent = C.blue, fill = C.bg) {
  addShape(slide, "roundRect", { left: x, top: y, width: w, height: h }, fill, { fill: C.line, width: 1 });
  addShape(slide, "rect", { left: x, top: y, width: w, height: 5 }, accent, { fill: "#00000000", width: 0 });
  addText(slide, slideNo, value, x + 18, y + 18, w - 36, 38, {
    fontSize: 28,
    color: C.ink,
    bold: true,
    typeface: FONT.title,
    role: "metric-value",
  });
  addText(slide, slideNo, label, x + 18, y + 62, w - 36, 28, {
    fontSize: 15,
    color: C.sub,
    role: "metric-label",
  });
}

function addPanel(slide, x, y, w, h, fill = C.bg) {
  return addShape(slide, "roundRect", { left: x, top: y, width: w, height: h }, fill, { fill: C.line, width: 1 });
}

function addColumnCard(slide, slideNo, x, y, w, h, title, bodyLines, accent = C.blue) {
  addPanel(slide, x, y, w, h, C.bg);
  addShape(slide, "rect", { left: x, top: y, width: w, height: 5 }, accent, { fill: "#00000000", width: 0 });
  addText(slide, slideNo, title, x + 18, y + 20, w - 36, 28, {
    fontSize: 18,
    color: C.ink,
    bold: true,
    typeface: FONT.title,
    role: "card-title",
  });
  addBulletList(slide, slideNo, bodyLines, x + 18, y + 70, w - 36, 54, 17, "card-bullets");
}

function addProcessStrip(slide, slideNo, x, y, w, label, body, index, accent = C.blue) {
  addPanel(slide, x, y, w, 98, index % 2 === 0 ? C.graySoft : C.bg);
  addText(slide, slideNo, String(index).padStart(2, "0"), x + 16, y + 18, 44, 28, {
    fontSize: 24,
    color: accent,
    bold: true,
    typeface: FONT.title,
    role: "strip-index",
  });
  addText(slide, slideNo, label, x + 84, y + 18, 220, 26, {
    fontSize: 16,
    color: C.ink,
    bold: true,
    typeface: FONT.title,
    role: "strip-label",
  });
  addText(slide, slideNo, body, x + 84, y + 48, w - 104, 34, {
    fontSize: 15,
    color: C.sub,
    typeface: FONT.body,
    role: "strip-body",
  });
}

function addProofGrid(slide, slideNo, x, y, w, h) {
  addPanel(slide, x, y, w, h, C.bg);
  const pad = 30;
  const gx = x + pad;
  const gy = y + 34;
  const gw = w - pad * 2;
  const gh = h - 84;

  addText(slide, slideNo, "2d", gx + gw / 2 - 18, gy + gh + 18, 40, 18, {
    fontSize: 13,
    color: C.sub,
    typeface: FONT.mono,
    align: "center",
    role: "proof-axis",
  });
  addText(slide, slideNo, "d", gx - 24, gy + gh / 2 - 10, 20, 18, {
    fontSize: 13,
    color: C.sub,
    typeface: FONT.mono,
    align: "right",
    role: "proof-axis",
  });

  addShape(slide, "rect", { left: gx, top: gy, width: gw, height: gh }, "#FFFFFF", { fill: C.ink, width: 1.2 });
  addShape(slide, "rect", { left: gx + gw / 2, top: gy, width: 1, height: gh }, C.line, { fill: "#00000000", width: 0 });
  addShape(slide, "rect", { left: gx, top: gy + gh / 3, width: gw, height: 1 }, C.line, { fill: "#00000000", width: 0 });
  addShape(slide, "rect", { left: gx, top: gy + (2 * gh) / 3, width: gw, height: 1 }, C.line, { fill: "#00000000", width: 0 });

  const pts = [
    [gx + gw * 0.18, gy + gh * 0.18],
    [gx + gw * 0.72, gy + gh * 0.16],
    [gx + gw * 0.22, gy + gh * 0.49],
    [gx + gw * 0.68, gy + gh * 0.50],
    [gx + gw * 0.28, gy + gh * 0.82],
    [gx + gw * 0.79, gy + gh * 0.80],
  ];
  pts.forEach(([px, py]) => {
    addShape(slide, "ellipse", { left: px - 5, top: py - 5, width: 10, height: 10 }, C.blue, { fill: "#FFFFFF", width: 1 });
  });

  addText(slide, slideNo, "最多容纳 6 个候选点", x + 24, y + h - 34, w - 48, 18, {
    fontSize: 14,
    color: C.green,
    bold: true,
    typeface: FONT.title,
    align: "center",
    role: "proof-caption",
  });
}

function addProofStep(slide, slideNo, x, y, idx, title, body, accent = C.blue) {
  addText(slide, slideNo, String(idx), x, y, 20, 20, {
    fontSize: 15,
    color: accent,
    bold: true,
    typeface: FONT.mono,
    role: "proof-step-index",
  });
  addText(slide, slideNo, title, x + 26, y - 2, 160, 20, {
    fontSize: 15,
    color: C.ink,
    bold: true,
    typeface: FONT.title,
    role: "proof-step-title",
  });
  addText(slide, slideNo, body, x + 26, y + 18, 520, 38, {
    fontSize: 14,
    color: C.sub,
    typeface: FONT.body,
    role: "proof-step-body",
  });
}

async function buildSlides() {
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  const benchmarkData = {
    categories: ["10000", "20000", "30000", "40000", "50000"],
    divide: [3.153, 7.156, 11.51, 14.568, 19.91],
    brute: [29.216, 128.086, 292.893, 519.692, 790.49],
  };

  const processPng = path.join(ROOT, "results", "process_dashboard.png");
  const processGif = path.join(ROOT, "results", "process_dashboard.gif");
  const linearPng = path.join(ROOT, "results", "benchmark_linear.png");
  const loglogPng = path.join(ROOT, "results", "benchmark_loglog.png");

  {
    const slideNo = 1;
    const slide = presentation.slides.add();
    slide.background.fill = C.bg;
    addShape(slide, "rect", { left: 72, top: 88, width: 10, height: 246 }, C.blue, { fill: "#00000000", width: 0 });
    addText(slide, slideNo, "平面最近点对问题", 104, 94, 520, 58, {
      fontSize: 42,
      color: C.ink,
      bold: true,
      typeface: FONT.title,
      role: "cover-title-main",
    });
    addText(slide, slideNo, "分治算法实验汇报", 104, 150, 420, 42, {
      fontSize: 22,
      color: C.sub,
      typeface: FONT.title,
      role: "cover-title-sub",
    });
    addStatement(slide, slideNo, "核心结论：在 10000 至 50000 点的全实测范围内，分治法相对蛮力法的最高速度提升达到 39.70 倍。", 104, 236, 690, C.green);
    addMiniCard(slide, slideNo, 104, 352, 196, 110, "O(n log n)", "论文主线模型", C.blue);
    addMiniCard(slide, slideNo, 322, 352, 196, 110, "39.70x", "最大实测速比", C.green, C.greenSoft);
    addMiniCard(slide, slideNo, 540, 352, 196, 110, "轨迹导出", "过程可视化链路", C.blue, C.blueSoft);
    addText(slide, slideNo, "汇报聚焦：论文模型、算法实现与优化、几何证明、实验结果分析", 104, 506, 760, 30, {
      fontSize: 16,
      color: C.sub,
      role: "cover-scope",
    });
    addTopSystem(slide, slideNo, "实验汇报", slideNo);
    await addImage(slide, slideNo, processPng, { left: 842, top: 122, width: 366, height: 440 }, "cover-process", "contain");
    slide.speakerNotes.setText("约35秒。开场先给出实测结论，再说明本次汇报仅聚焦论文模型、算法实现与优化、几何证明和实验结果分析。");
  }

  {
    const slideNo = 2;
    const slide = presentation.slides.add();
    slide.background.fill = C.bg;
    addTopSystem(slide, slideNo, "论文模型", slideNo);
    addTitleBlock(slide, slideNo, "论文模型与实验定位", "本页仅说明理论模型来源以及本实验的工程化扩展边界");

    addStatement(slide, slideNo, "本实验采用 Shamos-Hoey 的经典确定性分治框架，并进一步构建了可验证、可导出、可可视化的完整实验链。", 72, 210, 1136, C.blue);

    addProcessStrip(slide, slideNo, 72, 312, 548, "理论模型", "确定性分治主线，目标时间复杂度为 O(n log n)。", 1, C.blue);
    addProcessStrip(slide, slideNo, 72, 426, 548, "工程实现", "完成 C++ 求解、正确性校验与 benchmark 数据导出。", 2, C.green);
    addProcessStrip(slide, slideNo, 72, 540, 548, "展示层", "输出 trace、steps、GIF、dashboard 以及实验报告。", 3, C.blue);

    addPanel(slide, 668, 312, 540, 326, C.graySoft);
    addText(slide, slideNo, "本实验的工作边界", 700, 340, 220, 26, {
      fontSize: 18,
      color: C.ink,
      bold: true,
      typeface: FONT.title,
      role: "boundary-title",
    });
    addBulletList(
      slide,
      slideNo,
      [
        "不展开蛮力法常识与基本定义。",
        "重点展示带状区域筛选、可视化导出与实验验证。",
        "全部结论均基于当前仓库中的真实实验输出。"
      ],
      700,
      392,
      450,
      58,
      18,
      "boundary-bullets",
    );
    slide.speakerNotes.setText("约40秒。先说明理论模型来源于经典论文，再说明本实验的核心工作是将该模型工程化，并明确汇报范围。");
  }

  {
    const slideNo = 3;
    const slide = presentation.slides.add();
    slide.background.fill = C.bg;
    addTopSystem(slide, slideNo, "算法实现", slideNo);
    addTitleBlock(slide, slideNo, "算法实现与关键优化", "本页聚焦三项直接影响性能、复现性与可展示性的设计");

    addColumnCard(
      slide,
      slideNo,
      72,
      244,
      360,
      334,
      "预排序与递归划分",
      [
        "点集分别按 x、y 排序，递归过程中仅进行线性划分。",
        "递归基限定为 |S|≤3，以控制边界处理复杂度。",
        "左右子问题规模保持均衡，递归深度稳定在 log n。"
      ],
      C.blue,
    );
    addColumnCard(
      slide,
      slideNo,
      460,
      244,
      360,
      334,
      "带状区域筛选",
      [
        "合并阶段仅搜索满足 |x-xm|<d 的 strip 区域。",
        "strip 按 y 排序后，利用 y 差进行快速剪枝。",
        "二维搜索由此压缩为常数级候选比较。"
      ],
      C.green,
    );
    addColumnCard(
      slide,
      slideNo,
      848,
      244,
      360,
      334,
      "可视化友好导出",
      [
        "trace_demo.csv 记录当前最优值被更新的时刻。",
        "steps_demo.csv 记录递归深度、中线位置与 strip 边界。",
        "这些数据进一步驱动 GIF、dashboard 与实验报告。"
      ],
      C.blue,
    );
    slide.speakerNotes.setText("约50秒。依次说明预排序与递归结构、strip 合并剪枝，以及为后续展示设计的 trace 与 steps 导出机制。");
  }

  {
    const slideNo = 4;
    const slide = presentation.slides.add();
    slide.background.fill = C.bg;
    addTopSystem(slide, slideNo, "几何证明", slideNo);
    addTitleBlock(slide, slideNo, "分治法合并步骤线性效率的几何证明", "报告附录中的核心结论：按 y 排序后，每个点至多只需比较后续 6 个点");

    addMiniCard(slide, slideNo, 72, 210, 186, 98, "2d × d", "候选比较矩形", C.blue, C.blueSoft);
    addMiniCard(slide, slideNo, 280, 210, 186, 98, "6", "最多容纳点数", C.green, C.greenSoft);
    addMiniCard(slide, slideNo, 488, 210, 186, 98, "< d", "小矩形对角线", C.blue, C.blueSoft);

    addPanel(slide, 72, 336, 632, 300, C.bg);
    addText(slide, slideNo, "证明思路", 98, 364, 140, 24, {
      fontSize: 18,
      color: C.ink,
      bold: true,
      typeface: FONT.title,
      role: "proof-title",
    });
    addProofStep(slide, slideNo, 98, 404, 1, "候选区域", "单点仅需考察分界线右侧的 2d×d 候选区域。", C.blue);
    addProofStep(slide, slideNo, 98, 462, 2, "区域切割", "将该矩形划分为 2×3 共 6 个小矩形，每块尺寸为 d/2 × d/3。", C.green);
    addProofStep(slide, slideNo, 98, 520, 3, "对角线界", "小矩形对角线长度为 d√13/6≈0.6009d，因此严格小于 d。", C.blue);
    addProofStep(slide, slideNo, 98, 578, 4, "鸽巢反证", "若候选区中出现至少 7 个点，则必有两点落入同一小矩形，其距离小于 d，与前提矛盾。", C.green);
    addText(slide, slideNo, "结论：每个点仅需与后续最多 6 个点比较，因此单层合并复杂度为 O(n)。", 98, 612, 560, 20, {
      fontSize: 15,
      color: C.green,
      bold: true,
      typeface: FONT.title,
      role: "proof-conclusion",
    });

    addProofGrid(slide, slideNo, 742, 210, 466, 426);
    addText(slide, slideNo, "文献依据：Shamos & Hoey, FOCS 1975", 820, 650, 310, 18, {
      fontSize: 12,
      color: C.sub,
      typeface: FONT.mono,
      role: "proof-citation",
    });
    slide.speakerNotes.setText("约55秒。依次说明候选区域、区域切割、对角线界和鸽巢反证，最后给出每点至多比较后续 6 个点的结论。");
  }

  {
    const slideNo = 5;
    const slide = presentation.slides.add();
    slide.background.fill = C.bg;
    addTopSystem(slide, slideNo, "过程可视化", slideNo);
    addTitleBlock(slide, slideNo, "过程可视化：不仅给出结果，也给出执行轨迹", "本页展示为答辩表达而设计的 GIF 与 dashboard 输出");

    addPanel(slide, 72, 212, 744, 430, C.bg);
    await addImage(slide, slideNo, processGif, { left: 92, top: 232, width: 704, height: 390 }, "process-gif", "contain");

    addPanel(slide, 846, 212, 362, 188, C.graySoft);
    await addImage(slide, slideNo, processPng, { left: 862, top: 226, width: 330, height: 156 }, "process-dashboard", "contain");

    addPanel(slide, 846, 424, 362, 218, C.bg);
    addBulletList(
      slide,
      slideNo,
      [
        "GIF 适合展示递归推进与最优点对更新过程。",
        "dashboard 同时保留全局视图、距离曲线与局部放大。",
        "因此 strip、中线与当前最优解能够被直接解释。"
      ],
      872,
      456,
      300,
      58,
      17,
      "viz-bullets",
    );
    slide.speakerNotes.setText("约55秒。先用左侧 GIF 展示执行过程，再用右上角静态图补充 dashboard 的三层信息：全局点云、距离下降曲线和局部放大。");
  }

  {
    const slideNo = 6;
    const slide = presentation.slides.add();
    slide.background.fill = C.bg;
    addTopSystem(slide, slideNo, "实验结果", slideNo);
    addTitleBlock(slide, slideNo, "实验结果：全实测数据支撑复杂度结论", "一张原生图表展示绝对时间差，一张 log-log 图验证增长阶与理论是否一致");

    addMiniCard(slide, slideNo, 72, 206, 206, 102, "9.27x", "10000 点速度提升", C.blue, C.blueSoft);
    addMiniCard(slide, slideNo, 294, 206, 206, 102, "39.70x", "50000 点速度提升", C.green, C.greenSoft);
    addMiniCard(slide, slideNo, 516, 206, 206, 102, "19.91ms", "50000 点分治均值", C.blue, C.bg);

    addPanel(slide, 72, 338, 742, 306, C.bg);
    const chart = slide.charts.add("line");
    chart.position = { left: 94, top: 364, width: 700, height: 252 };
    chart.title = "运行时间对比（均值，ms）";
    chart.categories = benchmarkData.categories;
    chart.hasLegend = true;
    chart.legend.position = "top";
    chart.titleTextStyle.fontSize = 16;
    chart.titleTextStyle.bold = true;
    chart.titleTextStyle.typeface = FONT.title;
    chart.titleTextStyle.fill = C.ink;
    chart.legend.textStyle.fontSize = 12;
    chart.legend.textStyle.typeface = FONT.body;
    chart.legend.textStyle.fill = C.sub;
    chart.xAxis.textStyle.fontSize = 11;
    chart.xAxis.textStyle.typeface = FONT.body;
    chart.xAxis.textStyle.fill = C.sub;
    chart.yAxis.textStyle.fontSize = 11;
    chart.yAxis.textStyle.typeface = FONT.body;
    chart.yAxis.textStyle.fill = C.sub;
    chart.lineOptions.grouping = "standard";
    chart.plotAreaFill = C.bg;
    const divideSeries = chart.series.add("分治法");
    divideSeries.values = benchmarkData.divide;
    divideSeries.categories = benchmarkData.categories;
    divideSeries.stroke = { width: 2.4, style: "solid", fill: C.blue };
    divideSeries.fill = C.blue;
    const bruteSeries = chart.series.add("蛮力法");
    bruteSeries.values = benchmarkData.brute;
    bruteSeries.categories = benchmarkData.categories;
    bruteSeries.stroke = { width: 2.4, style: "solid", fill: C.red };
    bruteSeries.fill = C.red;

    addPanel(slide, 846, 206, 362, 438, C.graySoft);
    await addImage(slide, slideNo, loglogPng, { left: 864, top: 232, width: 326, height: 224 }, "loglog", "contain");
    addStatement(slide, slideNo, "在 log-log 坐标下，分治曲线更接近 n log n，而蛮力曲线更接近 n²。", 868, 490, 314, C.green);
    addText(slide, slideNo, "因此，本页不仅比较绝对耗时，也检验实测增长阶是否与理论分析一致。", 868, 566, 300, 44, {
      fontSize: 15,
      color: C.sub,
      role: "results-note",
    });
    slide.speakerNotes.setText("约60秒。先说明左图对应 5 次重复实验后的均值；在全部测试规模下，分治法均显著快于蛮力法。再看右侧 log-log 图，说明实测增长趋势与理论复杂度一致。");
  }

  {
    const slideNo = 7;
    const slide = presentation.slides.add();
    slide.background.fill = C.bg;
    addTopSystem(slide, slideNo, "结论", slideNo);
    addTitleBlock(slide, slideNo, "结论与收束", "最后 30 至 40 秒，仅保留三条结论与一句结束语");

    addStatement(slide, slideNo, "本实验的价值不仅在于完成最近点对算法实现，更在于将理论分析、可视化表达与实验验证整合为一条完整链路。", 72, 210, 1080, C.blue);

    addProcessStrip(slide, slideNo, 72, 324, 612, "性能结论", "在 10000 至 50000 点的全实测范围内，分治法始终显著快于蛮力法。", 1, C.blue);
    addProcessStrip(slide, slideNo, 72, 438, 612, "表达结论", "trace、steps、GIF 与 dashboard 使分治过程可以被清晰解释，而不仅仅给出最终结果。", 2, C.green);
    addProcessStrip(slide, slideNo, 72, 552, 612, "实验价值", "复杂度结论获得真实实验数据支撑，整体实现适合作为课程实验展示版本。", 3, C.blue);

    await addImage(slide, slideNo, linearPng, { left: 732, top: 304, width: 476, height: 264 }, "linear-benchmark", "contain");
    addPanel(slide, 776, 586, 388, 72, C.greenSoft);
    addText(slide, slideNo, "一句话结束：实现正确，机制可解释，结论可信。", 794, 608, 352, 26, {
      fontSize: 18,
      color: C.ink,
      bold: true,
      typeface: FONT.title,
      align: "center",
      role: "closing-line",
    });
    slide.speakerNotes.setText("约35秒。最后收束三点：性能优势、可视化价值、实验链完整性。结束语为：实现正确，机制可解释，结论可信。");
  }

  return presentation;
}

async function saveBlobToFile(blob, filePath) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  await fs.writeFile(filePath, bytes);
}

async function exportDeck(presentation) {
  const inspect = [
    { kind: "deck", slideCount: presentation.slides.count, slideSize: { width: W, height: H } },
    ...inspectRecords,
  ];
  await fs.writeFile(INSPECT_PATH, inspect.map((x) => JSON.stringify(x)).join("\n") + "\n", "utf8");

  for (let i = 0; i < presentation.slides.items.length; i += 1) {
    const slide = presentation.slides.items[i];
    const preview = await presentation.export({ slide, format: "png", scale: 1 });
    await saveBlobToFile(preview, path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`));
  }

  const pptxBlob = await PresentationFile.exportPptx(presentation);
  const pptxPath = path.join(OUT_DIR, "output.pptx");
  await pptxBlob.save(pptxPath);
  return pptxPath;
}

await ensureDirs();
const presentation = await buildSlides();
const pptxPath = await exportDeck(presentation);
console.log(pptxPath);
