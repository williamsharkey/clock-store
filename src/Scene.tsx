import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { WebGPURenderer } from 'three/webgpu'
import {
  Fn, instanceIndex, attributeArray, uniform,
  float, vec3, cos, sin, sqrt, fract,
} from 'three/tsl'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Pane } from 'tweakpane'

const HIERARCHY = [
  { name: 'century', turnsPerParent: 1,   spt: 32 },
  { name: 'decade',  turnsPerParent: 10,  spt: 32 },
  { name: 'year',    turnsPerParent: 10,  spt: 32 },
  { name: 'day',     turnsPerParent: 365, spt: 32 },
  { name: 'hour',    turnsPerParent: 24,  spt: 16 },
  { name: 'minute',  turnsPerParent: 60,  spt: 16 },
  { name: 'second',  turnsPerParent: 60,  spt: 32 },
]
const WINDOW = 120
const TAU = Math.PI * 2

// ── CPU evalCoil ─────────────────────────────────────────────────────────────
function evalCoil(
  t: number, maxLevel: number,
  totals: number[], offsets: number[],
  R: number, L: number, omega: number, tMag: number,
  outPos: { x: number; y: number; z: number },
  outN?: { x: number; y: number; z: number },
  outB?: { x: number; y: number; z: number },
) {
  const theta = t * omega
  const cT = Math.cos(theta), sT = Math.sin(theta)
  let px = R * cT, py = (t - 0.5) * L, pz = R * sT
  let nx = -cT, ny = 0, nz = -sT
  let bx = -L * sT / tMag, by = -R * omega / tMag, bz = L * cT / tMag
  let dtx = -R * omega * sT, dty = L, dtz = R * omega * cT

  for (let lvl = 1; lvl <= maxLevel; lvl++) {
    const alpha = t * totals[lvl] * TAU
    const cA = Math.cos(alpha), sA = Math.sin(alpha)
    const off = offsets[lvl]
    const dx = cA * nx + sA * bx, dy = cA * ny + sA * by, dz = cA * nz + sA * bz
    const ex = -sA * nx + cA * bx, ey = -sA * ny + cA * by, ez = -sA * nz + cA * bz
    px += off * dx; py += off * dy; pz += off * dz
    const w = off * totals[lvl] * TAU
    dtx += w * ex; dty += w * ey; dtz += w * ez
    const tLen = Math.sqrt(dtx * dtx + dty * dty + dtz * dtz)
    const tx = dtx / tLen, ty = dty / tLen, tz = dtz / tLen
    const dot = dx * tx + dy * ty + dz * tz
    let nnx = dx - dot * tx, nny = dy - dot * ty, nnz = dz - dot * tz
    const nLen = Math.sqrt(nnx * nnx + nny * nny + nnz * nnz)
    nnx /= nLen; nny /= nLen; nnz /= nLen
    nx = nnx; ny = nny; nz = nnz
    bx = ty * nnz - tz * nny; by = tz * nnx - tx * nnz; bz = tx * nny - ty * nnx
  }
  outPos.x = px; outPos.y = py; outPos.z = pz
  if (outN) { outN.x = nx; outN.y = ny; outN.z = nz }
  if (outB) { outB.x = bx; outB.y = by; outB.z = bz }
}

// ── GPU compute ──────────────────────────────────────────────────────────────
function createCoilCompute(
  maxLevel: number, totalTurns: number[], maxPts: number,
  offUniforms: ReturnType<typeof uniform>[],
  tBaseU: ReturnType<typeof uniform>, tStepU: ReturnType<typeof uniform>,
  baseFracUniforms: ReturnType<typeof uniform>[],
  radiusU: ReturnType<typeof uniform>, lengthU: ReturnType<typeof uniform>,
  omegaU: ReturnType<typeof uniform>, tmagU: ReturnType<typeof uniform>,
  posBuf: ReturnType<typeof attributeArray>,
) {
  return Fn(() => {
    const idx = instanceIndex
    const t = tBaseU.add(tStepU.mul(idx.toFloat()))
    const helixFrac = baseFracUniforms[0].add(tStepU.mul(float(totalTurns[0])).mul(idx.toFloat()))
    const theta = fract(helixFrac).mul(float(TAU))
    const cT = cos(theta), sT = sin(theta)
    const px = radiusU.mul(cT).toVar('px')
    const py = t.sub(0.5).mul(lengthU).toVar('py')
    const pz = radiusU.mul(sT).toVar('pz')
    const nx = cT.negate().toVar('nx'), ny = float(0).toVar('ny'), nz = sT.negate().toVar('nz')
    const bx = lengthU.negate().mul(sT).div(tmagU).toVar('bx')
    const by = radiusU.negate().mul(omegaU).div(tmagU).toVar('by')
    const bz = lengthU.mul(cT).div(tmagU).toVar('bz')
    for (let lvl = 1; lvl <= maxLevel; lvl++) {
      const lvlFrac = baseFracUniforms[lvl].add(tStepU.mul(float(totalTurns[lvl])).mul(idx.toFloat()))
      const alpha = fract(lvlFrac).mul(float(TAU))
      const cA = cos(alpha), sA = sin(alpha)
      const off = offUniforms[lvl]
      const nnx = cA.mul(nx).add(sA.mul(bx)).toVar(`nnx${lvl}`)
      const nny = cA.mul(ny).add(sA.mul(by)).toVar(`nny${lvl}`)
      const nnz = cA.mul(nz).add(sA.mul(bz)).toVar(`nnz${lvl}`)
      const nbx = sA.negate().mul(nx).add(cA.mul(bx)).toVar(`nbx${lvl}`)
      const nby = sA.negate().mul(ny).add(cA.mul(by)).toVar(`nby${lvl}`)
      const nbz = sA.negate().mul(nz).add(cA.mul(bz)).toVar(`nbz${lvl}`)
      px.addAssign(off.mul(nnx)); py.addAssign(off.mul(nny)); pz.addAssign(off.mul(nnz))
      nx.assign(nnx); ny.assign(nny); nz.assign(nnz)
      bx.assign(nbx); by.assign(nby); bz.assign(nbz)
    }
    posBuf.element(idx).assign(vec3(px, py, pz))
  })().compute(maxPts)
}

// ── Types ────────────────────────────────────────────────────────────────────
interface LevelGPU {
  posBuf: ReturnType<typeof attributeArray>; compute: ReturnType<typeof createCoilCompute>
  line: THREE.Line; maxPts: number
}
interface MarkerTier {
  segments: THREE.LineSegments; posAttr: THREE.BufferAttribute
  level: number; yearStep: number; count: number; lineLength: number
  skipEvery: number; maxDist: number; baseFontPx: number
}
interface Runtime {
  renderer: WebGPURenderer; scene: THREE.Scene
  camera: THREE.PerspectiveCamera; controls: OrbitControls
  levels: LevelGPU[]; tiers: MarkerTier[]; pane: Pane
}

export default function Scene() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayRef = useRef<HTMLDivElement>(null)
  const runtimeRef = useRef<Runtime | null>(null)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    const overlay = overlayRef.current
    if (!canvas || !overlay) return
    let disposed = false

    const START_YEAR = -10000, END_YEAR = 2100
    const TOTAL_YEARS = END_YEAR - START_YEAR
    const CENTURY_TURNS = TOTAL_YEARS / 100

    const params = {
      coilRadius: 1, turnSpacing: 10, fillFactor: 0.55,
      offsets: [0, 0, 0, 0, 0, 0, 0],
      panSpeed: 10, focusT: 0.5,
    }

    function syncOffsets() {
      let off = params.coilRadius
      for (let i = 1; i < params.offsets.length; i++) {
        off = off * params.fillFactor / Math.sqrt(HIERARCHY[i].turnsPerParent)
        params.offsets[i] = off
      }
    }
    syncOffsets()

    const tBaseU = uniform(0), tStepU = uniform(0)
    const baseFracUniforms = HIERARCHY.map(() => uniform(0))
    const radiusU = uniform(params.coilRadius)
    const lengthU = uniform(CENTURY_TURNS * params.turnSpacing)
    const omegaU = uniform(CENTURY_TURNS * TAU), tmagU = uniform(1)
    const offUniforms = params.offsets.map(v => uniform(v))

    function syncHelixUniforms() {
      const R = params.coilRadius, L = CENTURY_TURNS * params.turnSpacing, omega = CENTURY_TURNS * TAU
      radiusU.value = R; lengthU.value = L; omegaU.value = omega
      tmagU.value = Math.sqrt(R * R * omega * omega + L * L)
    }

    function getTotalTurns(): number[] {
      const t = [CENTURY_TURNS]
      for (let i = 1; i < HIERARCHY.length; i++) t.push(t[i - 1] * HIERARCHY[i].turnsPerParent)
      return t
    }

    function getHelixConsts() {
      const R = params.coilRadius, L = CENTURY_TURNS * params.turnSpacing, omega = CENTURY_TURNS * TAU
      return { R, L, omega, tMag: Math.sqrt(R * R * omega * omega + L * L) }
    }

    function winBounds(totalTurns: number, spt: number) {
      const halfT = (WINDOW / 2) / totalTurns
      const tS = Math.max(0, params.focusT - halfT), tE = Math.min(1, params.focusT + halfT)
      return { tS, tR: tE - tS }
    }

    // ── Camera ──────────────────────────────────────────────────────────────
    const _pos = { x: 0, y: 0, z: 0 }, _nrm = { x: 0, y: 0, z: 0 }, _bin = { x: 0, y: 0, z: 0 }
    const _rN = new THREE.Vector3(), _rB = new THREE.Vector3(), _tan = new THREE.Vector3(), _off = new THREE.Vector3()
    const _cfP0 = { x: 0, y: 0, z: 0 }, _cfP1 = { x: 0, y: 0, z: 0 }
    const _cfNv = new THREE.Vector3(), _cfBv = new THREE.Vector3(), _cfTmp = new THREE.Vector3()
    let prevFocusT = params.focusT
    let _cachedTotals: number[] | null = null
    function getCachedTotals() { if (!_cachedTotals) _cachedTotals = getTotalTurns(); return _cachedTotals }

    function cameraFrame(t: number) {
      const totals = getCachedTotals(), hc = getHelixConsts()
      evalCoil(t, 3, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, _pos, _nrm, _bin)
      const eps = 1e-7
      evalCoil(Math.max(0, t - eps), 3, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, _cfP0)
      evalCoil(Math.min(1, t + eps), 3, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, _cfP1)
      _tan.set(_cfP1.x - _cfP0.x, _cfP1.y - _cfP0.y, _cfP1.z - _cfP0.z).normalize()
      _cfNv.set(_nrm.x, _nrm.y, _nrm.z); _cfTmp.set(_nrm.x, _nrm.y, _nrm.z)
      _cfNv.addScaledVector(_tan, -_cfTmp.dot(_tan)).normalize()
      _cfBv.crossVectors(_tan, _cfNv)
      const da = t * getCachedTotals()[3] * TAU, c = Math.cos(da), s = Math.sin(da)
      _rN.copy(_cfNv).multiplyScalar(c).addScaledVector(_cfBv, s)
      _rB.copy(_cfNv).multiplyScalar(-s).addScaledVector(_cfBv, c)
    }

    const LEVEL_HUES = [0, 0.08, 0.16, 0.33, 0.55, 0.72, 0.88]

    function createLevels(scene: THREE.Scene): LevelGPU[] {
      syncHelixUniforms()
      const totals = getTotalTurns()
      return HIERARCHY.map((h, lvl) => {
        const maxPts = WINDOW * h.spt + 1
        const posBuf = attributeArray(new Float32Array(maxPts * 3), 'vec3')
        const compute = createCoilCompute(lvl, totals, maxPts, offUniforms, tBaseU, tStepU, baseFracUniforms, radiusU, lengthU, omegaU, tmagU, posBuf)
        const col = new THREE.Color().setHSL(LEVEL_HUES[lvl], 1, 0.5)
        const mat = new THREE.LineBasicMaterial({ color: col })
        const geom = new THREE.BufferGeometry(); geom.setAttribute('position', posBuf.value)
        const line = new THREE.Line(geom, mat); line.frustumCulled = false; scene.add(line)
        return { posBuf, compute, line, maxPts }
      })
    }

    function disposeLevels(rt: Runtime) { for (const lv of rt.levels) { rt.scene.remove(lv.line); lv.line.geometry.dispose() }; rt.levels = [] }

    function dispatchCompute(rt: Runtime) {
      const totals = getTotalTurns(), hc = getHelixConsts(), pos = { x: 0, y: 0, z: 0 }
      for (let i = 0; i < rt.levels.length; i++) {
        const w = winBounds(totals[i], HIERARCHY[i].spt), maxPts = rt.levels[i].maxPts
        const arr = rt.levels[i].posBuf.value.array as Float32Array
        for (let v = 0; v < maxPts; v++) {
          const t = w.tS + w.tR * v / (maxPts - 1)
          evalCoil(t, i, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, pos)
          arr[v * 3] = pos.x; arr[v * 3 + 1] = pos.y; arr[v * 3 + 2] = pos.z
        }
        rt.levels[i].posBuf.value.needsUpdate = true
        ;(rt.levels[i].posBuf as any).version = (rt.levels[i].posBuf as any).version + 1 || 1
      }
    }

    const _ufP = new THREE.Vector3(), _ufP2 = new THREE.Vector3()

    function updateFocus(rt: Runtime) {
      const oldT = prevFocusT, newT = params.focusT; prevFocusT = newT
      cameraFrame(oldT); _ufP.set(_pos.x, _pos.y, _pos.z)
      _off.copy(rt.camera.position).sub(_ufP)
      const lx = _off.dot(_rN), ly = _off.dot(_rB), lz = _off.dot(_tan)
      dispatchCompute(rt)
      cameraFrame(newT); _ufP2.set(_pos.x, _pos.y, _pos.z)
      rt.camera.position.copy(_ufP2).addScaledVector(_rN, lx).addScaledVector(_rB, ly).addScaledVector(_tan, lz)
      rt.controls.target.copy(_ufP2)
    }

    // ── Date formatting ─────────────────────────────────────────────────────
    function focusTToDate(ft: number): string {
      const totalSeconds = TOTAL_YEARS * 365.25 * 24 * 3600
      const date = new Date(START_YEAR * 365.25 * 24 * 3600 * 1000 + ft * totalSeconds * 1000)
      const y = date.getUTCFullYear(), m = date.getUTCMonth() + 1, d = date.getUTCDate()
      const h = date.getUTCHours(), min = date.getUTCMinutes(), s = date.getUTCSeconds()
      const pad = (n: number) => String(n).padStart(2, '0')
      return `${y < 0 ? `${-y} BCE` : `${y} CE`} ${pad(m)}/${pad(d)} ${pad(h)}:${pad(min)}:${pad(s)}`
    }

    function yearLabel(year: number): string {
      return year < 0 ? `${-year} BCE` : year === 0 ? '1 BCE' : `${year} CE`
    }

    // ── Init ────────────────────────────────────────────────────────────────
    const init = async () => {
      if (!navigator.gpu) { document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#fff;background:#111;font-family:system-ui;font-size:1.5rem;">WebGPU not supported</div>'; return }
      const w = window.innerWidth, h = window.innerHeight
      const renderer = new WebGPURenderer({ canvas: canvas!, antialias: true })
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); renderer.setSize(w, h, false)
      await renderer.init()
      if (!(renderer as any).backend?.isWebGPUBackend) { renderer.dispose(); return }
      if (disposed) { renderer.dispose(); return }

      const scene = new THREE.Scene(); scene.background = new THREE.Color(0x111111)
      const camera = new THREE.PerspectiveCamera(60, w / h, 0.0001, 500)
      const controls = new OrbitControls(camera, renderer.domElement)
      controls.enableDamping = true; controls.dampingFactor = 0.08; controls.enablePan = false

      scene.add(new THREE.DirectionalLight(0xffffff, 2).translateX(3).translateY(5).translateZ(4))
      scene.add(new THREE.AmbientLight(0xffffff, 0.3))

      const pane = new Pane({ title: 'Coilendar' })

      // ── Marker tiers (lines only — labels are DOM) ──────────────────────
      function createMarkerTier(count: number, yearStep: number, lineLength: number, level: number, skipEvery: number, maxDist: number, baseFontPx: number): MarkerTier {
        const posArr = new Float32Array(count * 2 * 3)
        const posAttr = new THREE.BufferAttribute(posArr, 3)
        const geom = new THREE.BufferGeometry(); geom.setAttribute('position', posAttr)
        const segments = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ color: 0xffffff }))
        segments.frustumCulled = false; scene.add(segments)
        return { segments, posAttr, level, yearStep, count, lineLength, skipEvery, maxDist, baseFontPx }
      }

      const centuryTier = createMarkerTier(Math.ceil(TOTAL_YEARS / 100) + 1, 100, 0.3, 0, 0, Infinity, 24)
      const decadeTier = createMarkerTier(Math.ceil(TOTAL_YEARS / 10) + 1, 10, 0.15, 1, 100, 50, 18)
      const yearTier = createMarkerTier(10, 1, 0.08, 2, 10, 5, 14)
      const focusTier = createMarkerTier(1, 0, 0.1, 3, 0, Infinity, 14)
      const tiers = [focusTier, centuryTier, decadeTier, yearTier]

      // ── DOM label pool ──────────────────────────────────────────────────
      const POOL_SIZE = 80
      const labelPool: HTMLDivElement[] = []
      for (let i = 0; i < POOL_SIZE; i++) {
        const div = document.createElement('div')
        div.style.cssText = 'position:absolute;font-family:monospace;pointer-events:none;white-space:nowrap;display:none'
        overlay.appendChild(div)
        labelPool.push(div)
      }
      let poolIdx = 0
      const _projVec = new THREE.Vector3()

      let currentDecade = -Infinity
      let focusLabelText = ''

      const rt: Runtime = { renderer, scene, camera, controls, levels: [], tiers, pane }
      runtimeRef.current = rt

      rt.levels = createLevels(scene)
      dispatchCompute(rt)

      cameraFrame(params.focusT)
      const p = new THREE.Vector3(_pos.x, _pos.y, _pos.z)
      controls.target.copy(p)
      camera.position.copy(p).addScaledVector(_rN, -2).addScaledVector(_rB, 0.5)

      // ── Controls ──────────────────────────────────────────────────────────
      let dragging = false, dragOriginX = 0, dragDisplacement = 0, focusDragging = false
      const focusBinding = pane.addBinding(params, 'focusT', { min: 0, max: 1, step: 0.0001, label: 'focus' })
      focusBinding.on('change', () => { if (!focusDragging) updateFocus(rt) })
      canvas.addEventListener('pointerdown', (e) => { if (e.button === 2) { dragging = true; focusDragging = true; dragOriginX = e.clientX; dragDisplacement = 0; e.preventDefault() } })
      canvas.addEventListener('pointermove', (e) => { if (dragging) dragDisplacement = (e.clientX - dragOriginX) * 0.01 })
      const stopDrag = () => { dragging = false; focusDragging = false }
      canvas.addEventListener('pointerup', stopDrag); canvas.addEventListener('pointercancel', stopDrag)
      canvas.addEventListener('contextmenu', (e) => e.preventDefault())

      pane.addBinding(params, 'coilRadius', { min: 0.1, max: 5, step: 0.1, label: 'coil radius' }).on('change', () => { syncHelixUniforms(); dispatchCompute(rt) })
      pane.addBinding(params, 'turnSpacing', { min: 0.1, max: 20, step: 0.1, label: 'turn spacing' }).on('change', () => { syncHelixUniforms(); dispatchCompute(rt) })
      pane.addBinding(params, 'fillFactor', { min: 0.1, max: 1.5, step: 0.01, label: 'fill factor' }).on('change', () => { syncOffsets(); dispatchCompute(rt) })
      pane.addBinding(params, 'panSpeed', { min: 1, max: 3600, step: 1, label: 'sec/pixel' })

      // ── Animate ───────────────────────────────────────────────────────────
      const _camRight = new THREE.Vector3(), _camFwd = new THREE.Vector3()
      const _mp = { x: 0, y: 0, z: 0 }

      const animate = () => {
        if (disposed) return
        rafRef.current = requestAnimationFrame(animate)

        if (dragging && dragDisplacement !== 0) {
          const totals = getCachedTotals()
          params.focusT = THREE.MathUtils.clamp(params.focusT - dragDisplacement * params.panSpeed / totals[totals.length - 1], 0, 1)
          updateFocus(rt)
        }

        rt.controls.update()
        rt.camera.getWorldDirection(_camFwd)
        _camRight.crossVectors(_camFwd, rt.camera.up).normalize()

        const totals = getCachedTotals(), hc = getHelixConsts()
        const sw = window.innerWidth, sh = window.innerHeight
        const camPos = rt.camera.position

        // Reset label pool
        poolIdx = 0
        for (const d of labelPool) d.style.display = 'none'

        // Update current decade for year tier
        const focusYear = START_YEAR + params.focusT * TOTAL_YEARS
        currentDecade = Math.floor(focusYear / 10) * 10

        // Process each marker tier: update lines + assign DOM labels
        for (const tier of tiers) {
          const { posAttr, level, yearStep, count, lineLength, skipEvery, maxDist, baseFontPx } = tier
          const arr = posAttr.array as Float32Array
          const isFocusTier = tier === focusTier
          const isYearTier = tier === yearTier

          for (let i = 0; i < count; i++) {
            const j = i * 6
            let year: number
            let t: number

            if (isFocusTier) {
              year = 0; t = params.focusT
            } else if (isYearTier) {
              year = currentDecade + i
              if (year < START_YEAR || year > END_YEAR || year % 10 === 0) {
                arr[j] = arr[j+1] = arr[j+2] = arr[j+3] = arr[j+4] = arr[j+5] = 0; continue
              }
              t = (year - START_YEAR) / TOTAL_YEARS
            } else {
              year = START_YEAR + i * yearStep
              if (skipEvery > 0 && year % skipEvery === 0) {
                arr[j] = arr[j+1] = arr[j+2] = arr[j+3] = arr[j+4] = arr[j+5] = 0; continue
              }
              t = (year - START_YEAR) / TOTAL_YEARS
            }

            evalCoil(t, level, totals, params.offsets, hc.R, hc.L, hc.omega, hc.tMag, _mp)
            const sx = _mp.x, sy = _mp.y, sz = _mp.z
            arr[j] = sx; arr[j+1] = sy; arr[j+2] = sz
            arr[j+3] = sx + lineLength * _camRight.x
            arr[j+4] = sy + lineLength * _camRight.y
            arr[j+5] = sz + lineLength * _camRight.z

            // Distance-based label visibility
            const dx = sx - camPos.x, dy = sy - camPos.y, dz = sz - camPos.z
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
            if (dist > maxDist) continue

            // Project to screen
            _projVec.set(sx, sy, sz).project(rt.camera)
            if (!isFocusTier && _projVec.z > 1) continue
            const scrX = (_projVec.x * 0.5 + 0.5) * sw
            const scrY = (-_projVec.y * 0.5 + 0.5) * sh
            if (!isFocusTier && (scrX < -100 || scrX > sw + 100 || scrY < -50 || scrY > sh + 50)) continue

            // Claim a label from the pool
            if (poolIdx >= POOL_SIZE) continue
            const div = labelPool[poolIdx++]

            const opacity = maxDist === Infinity ? 1 : Math.max(0.2, 1 - dist / maxDist)
            const fontSize = isFocusTier ? baseFontPx : Math.min(baseFontPx, Math.max(6, baseFontPx * 2 / dist))
            const color = isFocusTier ? '#ff4444' : 'white'
            const text = isFocusTier ? focusTToDate(params.focusT) : yearLabel(year)

            div.textContent = text
            div.style.display = 'block'
            div.style.left = `${scrX}px`
            div.style.top = `${scrY}px`
            div.style.fontSize = `${fontSize}px`
            div.style.opacity = String(opacity)
            div.style.color = color
          }
          posAttr.needsUpdate = true
        }

        rt.renderer.render(rt.scene, rt.camera)
      }
      rafRef.current = requestAnimationFrame(animate)
    }

    init()
    const handleResize = () => {
      const rt = runtimeRef.current; if (!rt) return
      rt.renderer.setSize(window.innerWidth, window.innerHeight, false)
      rt.camera.aspect = window.innerWidth / window.innerHeight
      rt.camera.updateProjectionMatrix()
    }
    window.addEventListener('resize', handleResize)
    return () => {
      disposed = true; window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(rafRef.current)
      const rt = runtimeRef.current
      if (rt) { rt.pane.dispose(); rt.controls.dispose(); disposeLevels(rt); rt.renderer.dispose(); runtimeRef.current = null }
      // Clean up DOM labels
      while (overlay.firstChild) overlay.removeChild(overlay.firstChild)
    }
  }, [])

  return (
    <>
      <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
      <div ref={overlayRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', overflow: 'hidden' }} />
    </>
  )
}
