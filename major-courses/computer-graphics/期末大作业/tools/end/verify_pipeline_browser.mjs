// Exercise the actual viewer in an isolated Edge process; never attach to personal tabs.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(path.join(root, 'main/end/modeling/viewer/package.json'));
const {chromium} = require('playwright');
const [url, packageArg] = process.argv.slice(2);
if (!url || !packageArg) throw new Error('Usage: node verify_pipeline_browser.mjs <url> <package-dir>');
const packageDir = path.resolve(packageArg);
const manifestBytes = await fs.readFile(path.join(packageDir, 'manifest.json'));
const manifest = JSON.parse(manifestBytes);
const screenshots = path.join(packageDir, 'previews');
await fs.mkdir(screenshots, {recursive:true});
const edge = process.env.PIPELINE_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const browser = await chromium.launch({executablePath:edge, headless:true, args:['--enable-unsafe-swiftshader']});
const errors = [], requests = [], screenshotsList = [];
const near = (a,b,label) => {
  assert.equal(a.length,b.length,label);
  assert.ok(a.every((v,i)=>Number.isFinite(v)&&Math.abs(v-b[i])<1e-4),`${label}: ${a} != ${b}`);
};
let report;
try {
  const page = await browser.newPage({viewport:{width:1280,height:800},deviceScaleFactor:1});
  const manifestUrl=new URL(new URL(url).searchParams.get('asset'),url).href;
  const modelUrl=new URL(manifest.model,manifestUrl).href;
  const contextUrl=manifest.preview_context ? new URL(manifest.preview_context.model,manifestUrl).href : null;
  const loadedBodies={}, responseReads=[];
  page.on('response',res=>{
    const resource=new URL(res.url());
    if(res.ok() && (res.url()===manifestUrl || res.url()===modelUrl || res.url()===contextUrl || resource.pathname==='/' || /\.(js|css)$/.test(resource.pathname))) {
      responseReads.push(res.body().then(body=>{loadedBodies[res.url()]=crypto.createHash('sha256').update(body).digest('hex');}));
    }
  });
  page.on('pageerror', e=>errors.push(e.message));
  page.on('request', req=>requests.push(req.url()));
  await page.route('**/*', route => {
    const host = new URL(route.request().url()).hostname;
    return ['127.0.0.1','localhost'].includes(host) ? route.continue() : route.abort();
  });
  await page.goto(url, {waitUntil:'networkidle'});
  await page.waitForFunction(()=>window.__assetProbe?.ready || window.__assetProbe?.errors?.length, {timeout:30000});
  const initial = await page.evaluate(()=>({ready:window.__assetProbe.ready,errors:window.__assetProbe.errors,state:window.__assetProbe.inspect()}));
  assert.equal(initial.ready,true,JSON.stringify(initial.errors));
  assert.deepEqual(initial.errors,[]);
  const initialGrid = await page.locator('#grid-toggle').isChecked();
  await Promise.all(responseReads);
  assert.equal(loadedBodies[manifestUrl],crypto.createHash('sha256').update(manifestBytes).digest('hex'),'Browser must receive the exact local manifest');
  assert.equal(loadedBodies[modelUrl],manifest.files[manifest.model],'Browser must receive the exact exported GLB');
  if (contextUrl) {
    assert.equal(loadedBodies[contextUrl],manifest.files[manifest.preview_context.model],'Browser must receive the exact separate context');
    assert.equal(initial.state.context.visible,true);
    await page.locator('#context-toggle').uncheck();
    assert.equal(await page.evaluate(()=>window.__assetProbe.inspect().context.visible),false);
    await page.locator('#context-toggle').check();
  }
  assert.equal(initial.state.statistics.triangles,manifest.statistics.triangles);
  near(initial.state.bounds.min,manifest.bounds.min,'browser minimum');
  near(initial.state.bounds.max,manifest.bounds.max,'browser maximum');
  for (const a of initial.state.anchors) near(a.world,a.expected,`anchor ${a.id}`);
  assert.ok(initial.state.nodes.every(n=>n.found),'All semantic nodes must survive GLB export');
  const hierarchy = await page.evaluate(()=>window.__assetProbe.testHierarchy());
  assert.ok(hierarchy.ok && hierarchy.changed && hierarchy.restored,'Parent transform must propagate and restore');
  const cpp = JSON.parse(await fs.readFile(path.join(packageDir,'checks/cpp.json'),'utf8'));
  near(initial.state.bounds.min,cpp.bounds.min,'C++ / browser min');
  near(initial.state.bounds.max,cpp.bounds.max,'C++ / browser max');
  near(hierarchy.after,cpp.hierarchy.world_after,'C++ / browser posed descendant');
  for (const anchor of initial.state.anchors) near(anchor.world,cpp.anchors.find(a=>a.id===anchor.id).world,`C++ / browser anchor ${anchor.id}`);
  for (const view of ['hero','front','top', ...(manifest.views.walk ? ['walk'] : [])]) {
    await page.locator(`[data-view="${view}"]`).click();
    await page.waitForTimeout(150);
    assert.equal(await page.evaluate(()=>window.__assetProbe.inspect().view),view);
    const filename=`web-${view}.png`;
    await page.screenshot({path:path.join(screenshots,filename)}); screenshotsList.push(filename);
  }
  await page.locator('[data-view="hero"]').click();
  const closedCanvas = await page.locator('canvas').screenshot();
  const maxAngle=manifest.interaction.range_deg[1];
  await page.locator('#angle').focus();
  await page.locator('#angle').press('End');
  await page.waitForTimeout(150);
  const opened=await page.evaluate(()=>window.__assetProbe.inspect());
  assert.equal(opened.angle,maxAngle);
  const openCanvas=await page.locator('canvas').screenshot();
  assert.notDeepEqual(openCanvas,closedCanvas,'Hinge UI must change rendered pixels');
  await page.screenshot({path:path.join(screenshots,'web-open.png')}); screenshotsList.push('web-open.png');
  await page.locator('#wireframe-toggle').check();
  await page.waitForTimeout(100);
  assert.notDeepEqual(await page.locator('canvas').screenshot(),openCanvas,'Wireframe UI must affect rendering');
  await page.locator('#wireframe-toggle').uncheck();
  await page.locator('#grid-toggle').uncheck();
  await page.locator('#grid-toggle').check();
  await page.locator('#grid-toggle').setChecked(initialGrid);
  await page.locator('#hierarchy-test').click();
  assert.ok((await page.locator('#hierarchy-result').innerText()).length>0);
  await page.evaluate(()=>window.__assetProbe.setAngle(0));
  const frames = await page.evaluate(async()=>{
    const gaps=[]; let prev=0;
    for(let i=0;i<70;i++){const now=await new Promise(requestAnimationFrame);if(i>10)gaps.push(now-prev);prev=now;}
    gaps.sort((a,b)=>a-b); return {medianFrameMs:gaps[Math.floor(gaps.length/2)],p95FrameMs:gaps[Math.floor(gaps.length*.95)],samples:gaps.length};
  });
  await page.setViewportSize({width:390,height:844});
  await page.waitForTimeout(200);
  const mobile = await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,canvas:document.querySelector('canvas').getBoundingClientRect().toJSON()}));
  assert.ok(mobile.scroll<=mobile.width+1,'Mobile layout must not overflow horizontally');
  assert.ok(mobile.canvas.width>100 && mobile.canvas.height>100,'Mobile viewport must remain usable');
  await page.screenshot({path:path.join(screenshots,'web-mobile.png'),fullPage:true}); screenshotsList.push('web-mobile.png');
  const errorPage=await browser.newPage();
  await errorPage.goto(new URL('/?asset=/missing/manifest.json',url).href);
  await errorPage.waitForFunction(()=>window.__assetProbe?.errors.length>0);
  assert.equal(await errorPage.evaluate(()=>window.__assetProbe.ready),false);
  await errorPage.close();
  assert.deepEqual(errors,[],'Unexpected browser exceptions');
  const resources={};
  for (const [request,hash] of Object.entries(loadedBodies)) {
    const resource=new URL(request);
    if(resource.pathname==='/') resources['index.html']=hash;
    else if(/\.(js|css)$/.test(resource.pathname)) resources[resource.pathname.replace(/^\//,'')]=hash;
  }
  report={status:'passed',manifest_sha256:crypto.createHash('sha256').update(manifestBytes).digest('hex'),
    browser:await browser.version(),initial:initial.state,hierarchy,openedAngle:opened.angle,
    frames,mobile,screenshots:screenshotsList,requests:[...new Set(requests)],resources,errors,
    limits:'Headless browser timing is recorded, not a claim about interactive desktop FPS.'};
} finally {
  await browser.close();
}
await fs.mkdir(path.join(packageDir,'checks'),{recursive:true});
await fs.writeFile(path.join(packageDir,'checks/browser.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify({status:report.status,triangles:report.initial.statistics.triangles,
  anchors:report.initial.anchors.length,hierarchy:report.hierarchy.ok,screenshots:report.screenshots,frames:report.frames},null,2));
