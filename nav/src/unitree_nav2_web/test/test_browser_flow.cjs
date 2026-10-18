/* Real headless browser, isolated HTTP fixtures: no ROS and no robot commands.
 * PUPPETEER_MODULE=/tmp/.../node_modules/puppeteer-core node test_browser_flow.cjs
 * BROWSER_EXECUTABLE defaults to the locally installed Firefox.
 * Verified with puppeteer-core 23.0.0 and Firefox 128 on this ARM64 machine.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const os = require('node:os');
const puppeteer = require(process.env.PUPPETEER_MODULE || 'puppeteer-core');
const web = path.resolve(__dirname, '../web');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'nav-browser-'));
const actions = Object.fromEntries(['acquire', 'prepare', 'prepare_pose', 'model_set', 'policy', 'zero', 'stop', 'reset',
  'hand_toggle', 'hand_compensation', 'hand_edit', 'hand_load', 'hand_save', 'cart_camera_start',
  'cart_yaw_zero', 'pushcart_enable', 'pushcart_disable', 'cart_detach'].map(k => [k, true]));
const lower = [[-1.05,-.724,0,-1.57,-1.75,-1.57,-1.75],[-1.05,-1.05,-1.75,0,0,0,0]];
const upper = [[1.05,1.05,1.75,0,0,0,0],[1.05,.742,0,1.57,1.75,1.57,1.75]];
const state = { phase: 'SELECT_MAP', message: '离线界面测试', map_ready: false, localized: false,
  processing: false, motion_enabled: false, navigation_active: false, navigation_task: {state:'IDLE'},
  motion_gate:{can_enable: true, blockers: []}, navigation_speed:{mode:'auto',speed:.35,ready:true,pending:false},
  locomotion:{online:true,owned:true,armed:true,healthy:true,ready:false,stage:'WAITING',stage_text:'等待接管',
    mode:'control',velocity:[0,0,0],actions,prepare_pose:'original',prepare_pose_selectable:true,
    model_set:'v81',model_set_selectable:true,
    model_set_options:[{value:'v81',label:'60000 轮 Homie + V8.1 高层'},{value:'v82',label:'68000 轮 Homie + V8.2 高层'}],
    prepare_pose_options:[{value:'original',label:'原准备姿态'},{value:'arms_down',label:'双臂下垂'}],
    hands:{active:true,healthy:true,pose:'grasp',grasp_revision:0,grasp_dirty:false,lower,upper,
      grasp:lower.map((r,i)=>r.map((x,j)=>(x+upper[i][j])/2)),compensation_available:true},
    high_policy:{enabled:false,camera:{fresh:true,yaw_rad:.1},zero_calibration:{}},
    cart:{attached:false,epoch:2,valid:true,position:[1,0,0]}},
  navigation_reference:{ready:true,mode:'walk',epoch:2} };
const calls=[], errors=[], dialogs=[];
let stamp=0, sequence=0, offline=false, rejectMap=false, delayMs=0, browser;
const mapName='大厅 + 20% & "x".pcd';
const server = http.createServer(async (req,res)=>{
  const url=new URL(req.url,'http://localhost');
  const send=data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data));};
  if (url.pathname.startsWith('/api/')) {
    if (url.pathname==='/api/status') {if(offline){res.destroy();return;} state.sent_monotonic=++stamp;return send({ok:true,status:state});}
    if (url.pathname==='/api/maps') return send({ok:true,maps:[{name:mapName,size_mb:12}]});
    calls.push({path:url.pathname,query:Object.fromEntries(url.searchParams)});
    if (delayMs) await new Promise(r=>setTimeout(r,delayMs));
    if (url.pathname==='/api/map/process') {
      if(rejectMap) return send({ok:false,error:'测试：地图处理被拒绝'});
      state.map_ready=true;state.phase='WAITING_INITIAL_POSE';
    }
    if (url.pathname==='/api/initial_pose') {state.localized=true;state.phase='READY';}
    if (url.pathname==='/api/navigation/speed') Object.assign(state.navigation_speed,{mode:url.searchParams.get('mode'),speed:Number(url.searchParams.get('speed'))});
    if (url.pathname==='/api/motion') state.motion_enabled=url.searchParams.get('enabled')==='true';
    if (url.pathname==='/api/stop') {state.navigation_active=false;state.motion_enabled=false;}
    if (url.pathname==='/api/goal') state.navigation_active=true;
    if (url.pathname==='/api/locomotion') {
      const action=url.searchParams.get('action');
      const loco=state.locomotion, hands=loco.hands, high=loco.high_policy;
      if(action==='prepare_pose') loco.prepare_pose=url.searchParams.get('pose');
      if(action==='model_set') loco.model_set=url.searchParams.get('model_set');
      if(action==='prepare') {loco.stage='HOLDING';loco.prepare_pose_selectable=false;loco.model_set_selectable=false;loco.actions.model_set=false;}
      if(action==='acquire') loco.stage='ZERO';
      if(action==='policy') loco.stage='POLICY';
      if(action==='stop') loco.stage='DAMPED';
      if(action==='reset') loco.stage='WAITING';
      loco.ready=loco.stage==='POLICY';
      const id=++sequence;
      loco.cart_request={sequence:id,state:'queued'};
      setTimeout(()=>{
        if(action==='hand_edit') {const i=Number(url.searchParams.get('joint'));hands.grasp[Math.floor(i/7)][i%7]=Number(url.searchParams.get('angle_rad'));hands.grasp_revision++;}
        if(action==='hand_compensation') hands.compensation_enabled=url.searchParams.get('enabled')==='true';
        if(action==='hand_toggle') hands.pose=hands.pose==='grasp'?'zero':'grasp';
        if(action==='hand_save') hands.editor_message='抓握角度已保存到测试文件';
        if(action==='hand_load') hands.editor_message='已加载文件，正在应用抓握角度';
        if(action==='cart_yaw_zero') high.zero_calibration={status:'done',message:'航向零位标定完成'};
        if(action==='pushcart_enable') {high.enabled=true;loco.cart.attached=true;state.navigation_reference.mode='cart';}
        if(action==='pushcart_disable') high.enabled=false;
        if(action==='cart_detach') {loco.cart.attached=false;state.navigation_reference.mode='walk';}
        loco.cart_request={sequence:id,state:'handled'};
      },120);
      return send({ok:true,request_id:id});
    }
    return send({ok:true,navigation_speed:state.navigation_speed});
  }
  const name=url.pathname==='/'?'index.html':url.pathname.slice(1);
  if(!['index.html','app.js','style.css'].includes(name)){res.statusCode=404;return res.end();}
  res.setHeader('Content-Type',name.endsWith('.js')?'application/javascript':name.endsWith('.css')?'text/css':'text/html');
  // Keep the stub in the page realm: Firefox 128 rejects cross-realm access
  // to the class injected by BiDi's evaluateOnNewDocument preload script.
  const socketStub = 'window.WebSocket=class{constructor(){setTimeout(()=>this.onopen?.(),0);}send(){}close(){}};\n';
  res.end((name==='app.js'?socketStub:'') + fs.readFileSync(path.join(web,name),'utf8'));
});
(async()=>{
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  browser=await puppeteer.launch({browser:'firefox',protocol:'webDriverBiDi',executablePath:process.env.BROWSER_EXECUTABLE||'/usr/bin/firefox',headless:true,userDataDir:path.join(temporary,'profile')});
  const page=await browser.newPage(); await page.setViewport({width:1400,height:1000});
  page.on('pageerror',e=>errors.push(e.message));
  page.on('dialog',async d=>{dialogs.push(d.message());await d.accept();});
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const refresh=()=>page.evaluate(()=>refreshStatus());
  const disabled=selector=>page.$eval(selector,e=>e.disabled);
  const click=async selector=>{await page.click(selector);};
  const waitCall=async count=>{for(let i=0;i<100&&calls.length<=count;i++)await new Promise(r=>setTimeout(r,20));assert.ok(calls.length>count,`No request after click ${count}`);};
  const mutate=async(selector,value)=>page.$eval(selector,(e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));},value);
  await refresh();await page.waitForFunction(()=>document.querySelector('#mapSelect').options.length===1);
  assert.equal(await page.$eval('#mapSelect',e=>e.value),mapName);
  assert.equal(await disabled('#initialMode'),true);
  // Invalid map input is caught before discarding the existing display.
  await mutate('#resolution','0.01'); let count=calls.length;await click('#processMap');assert.equal(calls.length,count);
  await mutate('#resolution','0.03'); await click('#processMap');await waitCall(count);await refresh();
  assert.equal(calls.at(-1).query.name,mapName);
  await page.evaluate(()=>{layers.map={width:100,height:100,resolution:.1,origin:[0,0]};draw();});
  const drag=async button=>{
    await click(button); const box=await page.$eval('#mapCanvas',e=>{const b=e.getBoundingClientRect();return{x:b.x,y:b.y,w:b.width,h:b.height};});
    await page.mouse.move(box.x+box.w*.45,box.y+box.h*.5);await page.mouse.down();
    await page.mouse.move(box.x+box.w*.55,box.y+box.h*.5);await page.mouse.up();
  };
  assert.equal(await disabled('#initialMode'),true);
  assert.equal(await disabled('#toolbarInitial'),true);
  assert.deepEqual(await page.$$eval('.steps li',items=>items.map(e=>e.id)),['stepMap','stepLoco','stepLoc','stepMotion','stepNav']);
  state.initial_pose_requires_policy=false;await refresh();assert.equal(await disabled('#initialMode'),false);
  state.initial_pose_requires_policy=true;await refresh();assert.equal(await disabled('#initialMode'),true);
  // Each lifecycle control uses its intended action, including both preparation choices.
  for(const key of ['v82','v81']){
    await mutate('#locoModelSet',key);await page.waitForFunction(()=>modelSetPending===null);
    assert.equal(calls.at(-1).query.model_set,key);
    assert.equal(await page.$eval('#locoModelSet',e=>e.value),key);
  }
  for(const pose of ['arms_down','original']){await mutate('#locoPreparePose',pose);await page.waitForFunction(()=>preparationPending===null);assert.equal(calls.at(-1).query.pose,pose);}
  for(const [id,action] of [['locoAcquire','acquire'],['locoPrepare','prepare'],['locoPolicy','policy']]){
    count=calls.length;await click('#'+id);await waitCall(count);await page.waitForFunction(()=>lifecyclePending===null);assert.equal(calls.at(-1).query.action,action);
    if(action==='prepare') assert.equal(await disabled('#locoModelSet'),true);
    assert.equal(await disabled('#initialMode'),action!=='policy');
  }
  state.locomotion.decelerating=true;await refresh();assert.equal(await disabled('#initialMode'),true);
  state.locomotion.decelerating=false;await refresh();
  count=calls.length;await drag('#initialMode');await waitCall(count);await refresh();assert.equal(calls.at(-1).path,'/api/initial_pose');
  state.message='AMCL 定位稳定；运控进入策略运行后可启用运动门';await refresh();
  assert.equal(await page.$eval('#statusText',e=>e.textContent),'AMCL 定位稳定；可启用运动门');
  state.motion_gate={can_enable:false,blockers:['等待导航里程计']};await refresh();
  assert.equal(await page.$eval('#statusText',e=>e.textContent),'AMCL 定位稳定；运动门等待：等待导航里程计');
  state.motion_gate={can_enable:true,blockers:[]};await refresh();
  count=calls.length;await drag('#toolbarInitial');await waitCall(count);await refresh();assert.equal(calls.at(-1).query.replace,'true');
  for(const [id,action] of [['locoZero','zero'],['locoStop','stop'],['locoReset','reset']]){
    count=calls.length;await click('#'+id);await waitCall(count);await page.waitForFunction(()=>lifecyclePending===null);assert.equal(calls.at(-1).query.action,action);
  }
  state.locomotion.stage='POLICY';state.locomotion.ready=true;await refresh();
  await page.$eval('#cartHandEditor',e=>e.closest('details').open=true);
  for(let i=0;i<14;i++){
    const value=(lower[Math.floor(i/7)][i%7]+upper[Math.floor(i/7)][i%7])/2+.001;
    count=calls.length;await mutate('#cartJoint'+i,String(value));await waitCall(count);
    await page.waitForFunction(()=>cartPending===null&&!cartBusy);
    assert.equal(calls.at(-1).query.joint,String(i));assert.ok(Math.abs(Number(calls.at(-1).query.angle_rad)-value)<1e-12);
    assert.equal(await page.$eval('#cartJoint'+i,e=>e.validity.valid),true);
  }
  for(const action of ['hand_toggle','hand_load','hand_save','cart_camera_start','cart_yaw_zero','pushcart_enable','pushcart_disable','cart_detach']){
    count=calls.length;const selector=`[data-cart-action="${action}"]`;await click(selector);await waitCall(count);
    await page.$eval(selector,e=>e.click()); // Must not duplicate a still-pending UDP operation.
    await page.waitForFunction(()=>cartPending===null&&!cartBusy);
    assert.equal(calls.length,count+1);assert.equal(calls.at(-1).query.action,action);
  }
  count=calls.length;await click('#cartCompensation');await waitCall(count);await page.waitForFunction(()=>cartPending===null&&!cartBusy);
  assert.equal(calls.at(-1).query.enabled,'true');
  assert.match(await page.$eval('#cartZeroStatus',e=>e.textContent),/标定完成/);
  assert.match(await page.$eval('#cartHandStatus',e=>e.textContent),/已保存/);
  await mutate('#speedMode','fixed');await mutate('#speedValue','0.42');count=calls.length;await click('#applySpeed');await waitCall(count);
  await page.waitForFunction(()=>!speedRequestBusy);assert.equal(calls.at(-1).query.speed,'0.42');assert.equal(calls.at(-1).query.mode,'fixed');
  await mutate('#speedSlider','0.3');assert.equal(await page.$eval('#speedValue',e=>e.value),'0.30');
  count=calls.length;await click('#motionButton');await waitCall(count);await page.waitForFunction(()=>!motionBusy);assert.equal(state.motion_enabled,true);
  state.motion_gate={can_enable:false,blockers:['等待车体过滤后的扫描'],suspended:true};await refresh();
  assert.match(await page.$eval('#statusText',e=>e.textContent),/导航暂缓/);
  assert.equal(await page.$eval('#gateBlockers',e=>e.hidden),false);
  assert.equal(await disabled('#motionButton'),false); // Closing remains available during a gap.
  state.motion_gate={can_enable:true,blockers:[],suspended:false};await refresh();
  count=calls.length;await drag('#goalMode');await waitCall(count);await refresh();assert.equal(calls.at(-1).path,'/api/goal');
  assert.equal(await disabled('[data-cart-action="hand_toggle"]'),true);assert.equal(await disabled('#processMap'),true);
  count=calls.length;await click('#stopButton');await waitCall(count);await refresh();assert.equal(state.motion_enabled,false);
  state.motion_enabled=true;await refresh();count=calls.length;await drag('#toolbarGoal');await waitCall(count);await refresh();
  count=calls.length;await click('#motionButton');await waitCall(count);await page.waitForFunction(()=>!motionBusy);assert.equal(state.motion_enabled,false);
  state.navigation_active=false;await refresh();
  const savedReference=state.navigation_reference;
  state.navigation_reference={mode:'cart',ready:true,footprint:[[-1,-.4],[.6,-.4],[.6,.4],[-1,.4]],
    cart_model:{body:[[-.5,-.25],[.5,-.25],[.5,.25],[-.5,.25]],handle:[[-.49,-.25],[-.49,.25]],wheels:[]}};
  state.cart_pose={x:4,y:4,yaw:0};await refresh();
  const cartPixel=()=>page.evaluate(()=>{
    visibility.footprint=false;ctx.clearRect(0,0,canvas.width,canvas.height);drawCart(1);
    const p=worldToScreen(4.4,3.85);return Array.from(ctx.getImageData(Math.round(p[0]),Math.round(p[1]),1,1).data);
  });
  assert.ok((await cartPixel())[3]>0,'Cart body must be visible at its model position');
  state.cart_pose.yaw=Math.PI/2;await refresh();
  assert.equal((await cartPixel())[3],0,'Cart body must rotate with the measured map heading');
  state.navigation_reference=savedReference;delete state.cart_pose;await refresh();
  await page.evaluate(()=>{visibility.footprint=true;});
  for(const layer of ['grid','map','global','local','scan','globalPath','localPath','robot','cart','footprint','radar']) {
    await page.$eval('.displays',e=>e.open=true);await click(`[data-display="${layer}"]`);
    assert.equal(await page.evaluate(k=>visibility[k],layer),false);
  }
  await click('#panMode');await click('#focusRobot');await click('#sidebarToggle');await click('#sidebarToggle');
  // Keep displayed map/pose when an operation is rejected.
  rejectMap=true;await page.evaluate(()=>{robotPose={x:1,y:2,yaw:0};});count=calls.length;await click('#processMap');await waitCall(count);await page.waitForFunction(()=>!mapBusy);
  assert.equal(await page.evaluate(()=>robotPose.x),1);
  assert.ok(dialogs.some(t=>t.includes('地图处理被拒绝')));
  // Old status must not reopen controls, and a dead status stream must gray them.
  await page.evaluate(()=>acceptStatus({...status,sent_monotonic:-1,motion_enabled:true}));
  assert.equal(await page.evaluate(()=>status.motion_enabled),false);
  offline=true;await page.waitForFunction(()=>!statusFresh(),{timeout:6000});await page.waitForFunction(()=>document.querySelector('#locoPolicy').disabled);
  assert.equal(await disabled('[data-cart-action="pushcart_enable"]'),true);assert.equal(await disabled('#motionButton'),true);assert.equal(await disabled('#stopButton'),false);
  assert.equal(await disabled('#locoModelSet'),true);
  offline=false;await refresh();await page.setViewport({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  assert.deepEqual(errors,[]);
  console.log('Browser OK: all Nav controls, 14 hand joints, pending requests, errors, stale status, desktop and mobile layout');
})().catch(e=>{console.error(e.stack);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));fs.rmSync(temporary,{recursive:true,force:true});});
