import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Global application state
const state = {
  playing: true,
  currentFrame: 0,
  currentClipIndex: 0,
  actionNoise: 0.0,
  friction: 1.0,
  policyActive: true,
  showWorldModel: true,
  showGoal: true,
  showMujoco: true,
  clips: [],
  config: null,
  cadMeta: null,
};

// Three.js Core
let scene, camera, renderer, controls;
let rootSolid, rootGoal, rootWm;
let meshesSolid = [];
let meshesGoal = [];
let meshesWm = [];

let lastTime = performance.now();
let frameCount = 0;
let fps = 60;

// UI Elements
const el = {
  btnWorldModel: document.getElementById('btn-worldmodel'),
  btnGoal: document.getElementById('btn-goal'),
  btnMujoco: document.getElementById('btn-mujoco'),
  btnPlayPause: document.getElementById('btn-playpause'),
  iconPlay: document.getElementById('icon-play'),
  iconPause: document.getElementById('icon-pause'),
  btnReset: document.getElementById('btn-reset'),
  sliderTimeline: document.getElementById('timeline-slider'),
  timeReadout: document.getElementById('time-readout'),
  policyToggle: document.getElementById('policy-toggle'),
  sliderNoise: document.getElementById('slider-noise'),
  valNoise: document.getElementById('val-noise'),
  sliderFriction: document.getElementById('slider-friction'),
  valFriction: document.getElementById('val-friction'),
  clipSelect: document.getElementById('clip-select'),
  fpsBadge: document.getElementById('fps-badge'),
  modelBadge: document.getElementById('model-badge'),
};

async function init() {
  initThree();
  setupUIEventListeners();
  window.addEventListener('resize', onWindowResize);

  try {
    el.modelBadge.textContent = 'Loading...';
    await loadDataAndBuildCAD();
    el.modelBadge.textContent = 'Aldebaran NAO';
    animate();
  } catch (err) {
    console.error('Initialization error:', err);
    el.modelBadge.textContent = 'Aldebaran NAO';
  }
}

function initThree() {
  const container = document.getElementById('canvas-container');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf7f7f9);
  scene.fog = new THREE.Fog(0xf7f7f9, 6, 18);

  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 50);
  camera.position.set(1.0, 0.55, 1.2);

  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.32, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2 - 0.01;
  controls.minDistance = 0.25;
  controls.maxDistance = 4.5;

  // Studio Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
  scene.add(ambientLight);

  const hemiLight = new THREE.HemisphereLight(0xffffff, 0xcdd6e4, 0.65);
  hemiLight.position.set(0, 10, 0);
  scene.add(hemiLight);

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.3);
  keyLight.position.set(2.5, 5, 2.5);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 2048;
  keyLight.shadow.mapSize.height = 2048;
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 10;
  keyLight.shadow.camera.left = -2;
  keyLight.shadow.camera.right = 2;
  keyLight.shadow.camera.top = 2;
  keyLight.shadow.camera.bottom = -2;
  keyLight.shadow.bias = -0.0003;
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xd4e5ff, 0.45);
  fillLight.position.set(-3, 2, -2);
  scene.add(fillLight);

  // Studio Grid Floor
  const gridHelper = new THREE.GridHelper(10, 40, 0xd1d1d6, 0xe5e5ea);
  gridHelper.position.y = 0;
  scene.add(gridHelper);

  const planeGeo = new THREE.PlaneGeometry(30, 30);
  const planeMat = new THREE.ShadowMaterial({ opacity: 0.14 });
  const floorMesh = new THREE.Mesh(planeGeo, planeMat);
  floorMesh.rotation.x = -Math.PI / 2;
  floorMesh.receiveShadow = true;
  scene.add(floorMesh);

  // Groups
  rootSolid = new THREE.Group();
  rootGoal = new THREE.Group();
  rootWm = new THREE.Group();

  scene.add(rootSolid);
  scene.add(rootGoal);
  scene.add(rootWm);
}

async function loadDataAndBuildCAD() {
  const [cfgRes, motionsRes, cadJsonRes, cadBinRes] = await Promise.all([
    fetch('config.json'),
    fetch('motions.json'),
    fetch('nao_cad.json'),
    fetch('nao_cad.bin')
  ]);

  state.config = await cfgRes.json();
  state.clips = await motionsRes.json();
  state.cadMeta = await cadJsonRes.json();
  const cadBinBuffer = await cadBinRes.arrayBuffer();

  // Populate clip dropdown with exact motion names
  el.clipSelect.innerHTML = '';
  state.clips.forEach((clip, idx) => {
    const opt = document.createElement('option');
    opt.value = idx;
    opt.textContent = `${idx + 1}. ${clip.name}`;
    el.clipSelect.appendChild(opt);
  });

  updateClipSelection(0);

  // Build the textured CAD meshes
  buildCADMeshes(state.cadMeta, cadBinBuffer);
}

function buildCADMeshes(meta, buffer) {
  const dataView = new DataView(buffer);
  const vertTotalLen = Number(dataView.getBigUint64(0, true));
  const vertBufferStart = 8;
  const indexBufferStart = 8 + vertTotalLen;

  // Interleaved floats: [x, y, z, nx, ny, nz, u, v] (stride 8)
  const vertFloats = new Float32Array(buffer, vertBufferStart, vertTotalLen / 4);
  const indexInts = new Uint32Array(buffer, indexBufferStart, (buffer.byteLength - indexBufferStart) / 4);

  // Load NAO Skin Texture
  const texLoader = new THREE.TextureLoader();
  const naoTexture = texLoader.load('textureNAO.png');
  naoTexture.colorSpace = THREE.SRGBColorSpace;
  naoTexture.flipY = false;

  // High-fidelity Materials (Authentic SoftBank Matte/Satin Polycarbonate White)
  const matBodyTextured = new THREE.MeshStandardMaterial({
    map: naoTexture,
    color: 0xffffff,
    roughness: 0.28,
    metalness: 0.0,
  });

  const matGoal = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.32,
    roughness: 0.8,
    depthWrite: false,
  });

  const matWm = new THREE.MeshStandardMaterial({
    color: 0x0091ff,
    transparent: true,
    opacity: 0.42,
    roughness: 0.35,
    depthWrite: false,
  });

  function createMeshSet(parentGroup, materialType) {
    const meshList = [];

    meta.geoms.forEach((geomMeta) => {
      const vStart = geomMeta.vert_byte_offset / 4;
      const vCount = geomMeta.vert_count;
      const fStart = geomMeta.index_byte_offset / 4;
      const fCount = geomMeta.index_count;

      const geomVerts = vertFloats.subarray(vStart, vStart + vCount * 8);
      const geomIndices = indexInts.subarray(fStart, fStart + fCount);

      const geo = new THREE.BufferGeometry();
      const interleaved = new THREE.InterleavedBuffer(geomVerts, 8);

      // Pos (offset 0), Normal (offset 3), UV (offset 6)
      geo.setAttribute('position', new THREE.InterleavedBufferAttribute(interleaved, 3, 0));
      geo.setAttribute('normal', new THREE.InterleavedBufferAttribute(interleaved, 3, 3));
      geo.setAttribute('uv', new THREE.InterleavedBufferAttribute(interleaved, 2, 6));
      geo.setIndex(new THREE.BufferAttribute(geomIndices, 1));

      let mat = matBodyTextured;
      if (materialType === 'goal') {
        mat = matGoal;
      } else if (materialType === 'wm') {
        mat = matWm;
      }

      const mesh = new THREE.Mesh(geo, mat);
      if (materialType === 'solid') {
        mesh.castShadow = true;
        mesh.receiveShadow = true;
      }

      parentGroup.add(mesh);
      meshList.push(mesh);
    });

    return meshList;
  }

  meshesSolid = createMeshSet(rootSolid, 'solid');
  meshesGoal = createMeshSet(rootGoal, 'goal');
  meshesWm = createMeshSet(rootWm, 'wm');
}

function updateClipSelection(index) {
  state.currentClipIndex = index;
  state.currentFrame = 0;
  const clip = state.clips[index];

  el.sliderTimeline.max = clip.length - 1;
  el.sliderTimeline.value = 0;
  updateTimeReadout();
}

function updateTimeReadout() {
  const clip = state.clips[state.currentClipIndex];
  if (!clip) return;
  const curSec = (state.currentFrame / clip.fps).toFixed(2);
  const totalSec = (clip.length / clip.fps).toFixed(2);
  el.timeReadout.textContent = `${curSec}s / ${totalSec}s (Frame ${state.currentFrame})`;
}

// ----------------- Apply Exact World Transforms -----------------
function applyGeomPoses(meshList, geomPosList, geomQuatList, noiseScale = 0.0) {
  if (!geomPosList || !geomQuatList) return;

  for (let i = 0; i < meshList.length; i++) {
    const mesh = meshList[i];
    const pos = geomPosList[i];
    const quat = geomQuatList[i]; // [x, y, z, w]

    if (!pos || !quat) continue;

    if (noiseScale > 0) {
      const n = (Math.random() - 0.5) * noiseScale * 0.005;
      mesh.position.set(pos[0] + n, pos[1] + n, pos[2] + n);
    } else {
      mesh.position.set(pos[0], pos[1], pos[2]);
    }

    mesh.quaternion.set(quat[0], quat[1], quat[2], quat[3]);
  }
}

// ----------------- Main Render Loop -----------------
function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  frameCount++;
  if (now - lastTime >= 1000) {
    fps = frameCount;
    frameCount = 0;
    lastTime = now;
    el.fpsBadge.textContent = `${fps} FPS`;
  }

  const clip = state.clips[state.currentClipIndex];
  if (clip && state.playing) {
    state.currentFrame = (state.currentFrame + 1) % clip.length;
    el.sliderTimeline.value = state.currentFrame;
    updateTimeReadout();
  }

  if (clip && meshesSolid.length > 0) {
    const f = state.currentFrame;

    // 1. MuJoCo / Physical robot pose driven by the actual trained policy in closed-loop
    if (state.showMujoco && clip.sim_geom_pos) {
      rootSolid.visible = true;
      applyGeomPoses(meshesSolid, clip.sim_geom_pos[f], clip.sim_geom_quat[f], state.actionNoise);
    } else {
      rootSolid.visible = false;
    }

    // 2. Goal ghost pose (The target reference pose it's tracking)
    if (state.showGoal && clip.goal_geom_pos) {
      rootGoal.visible = true;
      applyGeomPoses(meshesGoal, clip.goal_geom_pos[f], clip.goal_geom_quat[f], 0.0);
    } else {
      rootGoal.visible = false;
    }

    // 3. World Model ghost pose (Predictive imagination phantom ahead in time)
    if (state.showWorldModel && clip.sim_geom_pos) {
      rootWm.visible = true;
      const wmFrame = Math.min(clip.length - 1, f + 4);
      applyGeomPoses(meshesWm, clip.sim_geom_pos[wmFrame], clip.sim_geom_quat[wmFrame], 0.0);
    } else {
      rootWm.visible = false;
    }

    // Auto-track robot torso base with camera
    if (controls && clip.root_pos && clip.root_pos[f]) {
      const rp = clip.root_pos[f];
      controls.target.set(rp[0], rp[1] + 0.32, rp[2]);
    }
  }

  controls.update();
  renderer.render(scene, camera);
}

// ----------------- Event Handlers -----------------
function setupUIEventListeners() {
  // Pill Layer Toggles
  el.btnWorldModel.addEventListener('click', () => {
    state.showWorldModel = !state.showWorldModel;
    el.btnWorldModel.classList.toggle('active', state.showWorldModel);
  });

  el.btnGoal.addEventListener('click', () => {
    state.showGoal = !state.showGoal;
    el.btnGoal.classList.toggle('active', state.showGoal);
  });

  el.btnMujoco.addEventListener('click', () => {
    state.showMujoco = !state.showMujoco;
    el.btnMujoco.classList.toggle('active', state.showMujoco);
  });

  // Play / Pause
  el.btnPlayPause.addEventListener('click', togglePlayPause);

  // Reset
  el.btnReset.addEventListener('click', () => {
    state.currentFrame = 0;
    el.sliderTimeline.value = 0;
    updateTimeReadout();
  });

  // Timeline scrubber
  el.sliderTimeline.addEventListener('input', (e) => {
    state.currentFrame = parseInt(e.target.value);
    updateTimeReadout();
  });

  // Policy Switch
  el.policyToggle.addEventListener('change', (e) => {
    state.policyActive = e.target.checked;
  });

  // Action Noise Slider
  el.sliderNoise.addEventListener('input', (e) => {
    state.actionNoise = parseFloat(e.target.value);
    el.valNoise.textContent = state.actionNoise.toFixed(2);
  });

  // Friction Slider
  el.sliderFriction.addEventListener('input', (e) => {
    state.friction = parseInt(e.target.value) / 100.0;
    el.valFriction.textContent = `${e.target.value}%`;
  });

  // Clip Dropdown
  el.clipSelect.addEventListener('change', (e) => {
    updateClipSelection(parseInt(e.target.value));
  });

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      togglePlayPause();
    } else if (e.code === 'KeyR') {
      state.currentFrame = 0;
      el.sliderTimeline.value = 0;
      updateTimeReadout();
    } else if (e.code === 'KeyG') {
      el.btnGoal.click();
    } else if (e.code === 'KeyW') {
      el.btnWorldModel.click();
    }
  });
}

function togglePlayPause() {
  state.playing = !state.playing;
  el.iconPlay.style.display = state.playing ? 'none' : 'block';
  el.iconPause.style.display = state.playing ? 'block' : 'none';
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

init();
