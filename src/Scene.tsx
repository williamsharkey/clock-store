import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { WebGPURenderer } from 'three/webgpu'
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

// ── Analytical recursive coil evaluation ─────────────────────────────────────
// Evaluate position (and optionally frame) at parameter t through levels 0..maxLevel.
// Each level rotates the parent's (N,B) frame by the winding angle and offsets.
// No stored data, no parallel transport — pure math.

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

  // Level 0: century helix — position, frame, and frame derivatives
  let px = R * cT, py = (t - 0.5) * L, pz = R * sT
  let nx = -cT, ny = 0, nz = -sT
  let bx = -L * sT / tMag, by = -R * omega / tMag, bz = L * cT / tMag
  let dnx = omega * sT, dny = 0, dnz = -omega * cT
  let dbx = -L * omega * cT / tMag, dby = 0, dbz = -L * omega * sT / tMag
  // Un-normalized tangent derivative dP/dt (NOT divided by tMag)
  let dtx = -R * omega * sT, dty = L, dtz = R * omega * cT

  for (let lvl = 1; lvl <= maxLevel; lvl++) {
    const alpha = t * totals[lvl] * Math.PI * 2
    const ap = totals[lvl] * Math.PI * 2
    const cA = Math.cos(alpha), sA = Math.sin(alpha)
    const off = offsets[lvl]

    // Rotated offset direction and its perpendicular
    const ox = cA * nx + sA * bx, oy = cA * ny + sA * by, oz = cA * nz + sA * bz

    // Position
    px += off * ox; py += off * oy; pz += off * oz

    // Derivative of offset direction
    const ddx = -ap * sA * nx + cA * dnx + ap * cA * bx + sA * dbx
    const ddy = -ap * sA * ny + cA * dny + ap * cA * by + sA * dby
    const ddz = -ap * sA * nz + cA * dnz + ap * cA * bz + sA * dbz
    const dex = -ap * cA * nx - sA * dnx - ap * sA * bx + cA * dbx
    const dey = -ap * cA * ny - sA * dny - ap * sA * by + cA * dby
    const dez = -ap * cA * nz - sA * dnz - ap * sA * bz + cA * dbz

    // Accumulate un-normalized tangent: dP/dt += off * d(offset)/dt
    dtx += off * ddx; dty += off * ddy; dtz += off * ddz

    // Normalize tangent for frame construction
    const tLen = Math.sqrt(dtx * dtx + dty * dty + dtz * dtz)
    const tx = dtx / tLen, ty = dty / tLen, tz = dtz / tLen

    // Gram-Schmidt: project offset direction perpendicular to tangent → new N
    const dot = ox * tx + oy * ty + oz * tz
    let nnx = ox - dot * tx, nny = oy - dot * ty, nnz = oz - dot * tz
    const nLen = Math.sqrt(nnx * nnx + nny * nny + nnz * nnz)
    nnx /= nLen; nny /= nLen; nnz /= nLen

    // New B = T × N
    const nbx = ty * nnz - tz * nny, nby = tz * nnx - tx * nnz, nbz = tx * nny - ty * nnx

    // Update frame and derivatives for next level
    nx = nnx; ny = nny; nz = nnz
    bx = nbx; by = nby; bz = nbz
    dnx = ddx; dny = ddy; dnz = ddz
    dbx = dex; dby = dey; dbz = dez
  }

  outPos.x = px; outPos.y = py; outPos.z = pz
  if (outN) { outN.x = nx; outN.y = ny; outN.z = nz }
  if (outB) { outB.x = bx; outB.y = by; outB.z = bz }
}

// ── Types ────────────────────────────────────────────────────────────────────
interface Runtime {
  renderer: WebGPURenderer; scene: THREE.Scene
  camera: THREE.PerspectiveCamera; controls: OrbitControls
  lines: THREE.Line[]
  lineMaterial: THREE.LineBasicMaterial; pane: Pane
}

export default function Scene() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const runtimeRef = useRef<Runtime | null>(null)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    let disposed = false

    const params = {
      centuryTurns: 2,
      coilRadius: 1,
      turnSpacing: 10,
      offsets: [0, 0.1740, 0.0871, 0.0300, 0.0007, 0.0005, 0.0001],
      panSpeed: 10,
      focusT: 0.5,
    }

    function getTotalTurns(): number[] {
      const t = [params.centuryTurns]
      for (let i = 1; i < HIERARCHY.length; i++) t.push(t[i - 1] * HIERARCHY[i].turnsPerParent)
      return t
    }

    function getHelixConsts() {
      const R = params.coilRadius, L = params.centuryTurns * params.turnSpacing
      const omega = params.centuryTurns * Math.PI * 2
      const tMag = Math.sqrt(R * R * omega * omega + L * L)
      return { R, L, omega, tMag }
    }

    function winBounds(totalTurns: number, spt: number) {
      const halfT = (WINDOW / 2) / totalTurns
      const tS = Math.max(0, params.focusT - halfT)
      const tE = Math.min(1, params.focusT + halfT)
      const tR = tE - tS
      return { tS, tR, pts: Math.max(2, Math.ceil(tR * totalTurns * spt) + 1) }
    }

    // ── Camera frame from year level ──────────────────────────────────────
    const _pos = { x: 0, y: 0, z: 0 }
    const _nrm = { x: 0, y: 0, z: 0 }
    const _bin = { x: 0, y: 0, z: 0 }
    const _rN = new THREE.Vector3(), _rB = new THREE.Vector3()
    const _tan = new THREE.Vector3()
    const _off = new THREE.Vector3()
    let prevFocusT = params.focusT

    function cameraFrame(t: number) {
      const totals = getTotalTurns(), { R, L, omega, tMag } = getHelixConsts()
      evalCoil(t, 2, totals, params.offsets, R, L, omega, tMag, _pos, _nrm, _bin)

      // Tangent from finite diff
      const eps = 1e-7
      const p0 = { x: 0, y: 0, z: 0 }, p1 = { x: 0, y: 0, z: 0 }
      evalCoil(Math.max(0, t - eps), 2, totals, params.offsets, R, L, omega, tMag, p0)
      evalCoil(Math.min(1, t + eps), 2, totals, params.offsets, R, L, omega, tMag, p1)
      _tan.set(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z).normalize()

      // Normal: project _nrm perpendicular to tangent
      const nv = new THREE.Vector3(_nrm.x, _nrm.y, _nrm.z)
      nv.addScaledVector(_tan, -nv.dot(_tan)).normalize()
      const bv = new THREE.Vector3().crossVectors(_tan, nv)

      // Rotate by day angle
      const dayAngle = t * getTotalTurns()[3] * Math.PI * 2
      const c = Math.cos(dayAngle), s = Math.sin(dayAngle)
      _rN.copy(nv).multiplyScalar(c).addScaledVector(bv, s)
      _rB.copy(nv).multiplyScalar(-s).addScaledVector(bv, c)
    }

    function disposeLevels(rt: Runtime) {
      for (const l of rt.lines) { rt.scene.remove(l); l.geometry.dispose() }
      rt.lines = []
    }

    // ── Build all coil levels ──────────────────────────────────────────────
    function rebuildCoils(rt: Runtime) {
      disposeLevels(rt)
      const totals = getTotalTurns()
      const { R, L, omega, tMag } = getHelixConsts()
      const pos = { x: 0, y: 0, z: 0 }
      const tmpColor = new THREE.Color()

      for (let lvl = 0; lvl < HIERARCHY.length; lvl++) {
        const w = winBounds(totals[lvl], HIERARCHY[lvl].spt)
        const posArr = new Float32Array(w.pts * 3)
        const colArr = new Float32Array(w.pts * 3)
        const denom = w.pts - 1
        const colorCycles = lvl === 0 ? params.centuryTurns : totals[lvl - 1]

        for (let i = 0; i < w.pts; i++) {
          const t = w.tS + w.tR * i / denom
          evalCoil(t, lvl, totals, params.offsets, R, L, omega, tMag, pos)
          const j = i * 3
          posArr[j] = pos.x; posArr[j + 1] = pos.y; posArr[j + 2] = pos.z
          tmpColor.setHSL((t * colorCycles) % 1, 1, 0.5)
          colArr[j] = tmpColor.r; colArr[j + 1] = tmpColor.g; colArr[j + 2] = tmpColor.b
        }

        const geom = new THREE.BufferGeometry()
        geom.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
        geom.setAttribute('color', new THREE.BufferAttribute(colArr, 3))
        const line = new THREE.Line(geom, rt.lineMaterial)
        rt.scene.add(line)
        rt.lines.push(line)
      }
    }

    function updateFocus(rt: Runtime) {
      const oldT = prevFocusT, newT = params.focusT
      prevFocusT = newT

      cameraFrame(oldT)
      const p = new THREE.Vector3(_pos.x, _pos.y, _pos.z)
      _off.copy(rt.camera.position).sub(p)
      const lx = _off.dot(_rN), ly = _off.dot(_rB), lz = _off.dot(_tan)

      rebuildCoils(rt)

      cameraFrame(newT)
      const p2 = new THREE.Vector3(_pos.x, _pos.y, _pos.z)
      rt.camera.position.copy(p2).addScaledVector(_rN, lx).addScaledVector(_rB, ly).addScaledVector(_tan, lz)
      rt.controls.target.copy(p2)
    }

    // ── Init ───────────────────────────────────────────────────────────────
    const init = async () => {
      if (!navigator.gpu) {
        document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#fff;background:#111;font-family:system-ui;font-size:1.5rem;">WebGPU is not supported in this browser.</div>'
        return
      }
      const w = window.innerWidth, h = window.innerHeight
      const renderer = new WebGPURenderer({ canvas: canvas!, antialias: true })
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      renderer.setSize(w, h, false)
      await renderer.init()
      if (!(renderer as any).backend?.isWebGPUBackend) {
        document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#fff;background:#111;font-family:system-ui;font-size:1.5rem;">WebGPU backend failed to initialize.</div>'
        renderer.dispose(); return
      }
      if (disposed) { renderer.dispose(); return }

      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x111111)
      const camera = new THREE.PerspectiveCamera(60, w / h, 0.0001, 500)
      const controls = new OrbitControls(camera, renderer.domElement)
      controls.enableDamping = true; controls.dampingFactor = 0.08
      controls.enablePan = false

      scene.add(new THREE.DirectionalLight(0xffffff, 2).translateX(3).translateY(5).translateZ(4))
      scene.add(new THREE.AmbientLight(0xffffff, 0.3))

      const lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true })
      const pane = new Pane({ title: 'Coilendar' })

      const rt: Runtime = { renderer, scene, camera, controls, lines: [], lineMaterial, pane }
      runtimeRef.current = rt

      rebuildCoils(rt)
      cameraFrame(params.focusT)
      const p = new THREE.Vector3(_pos.x, _pos.y, _pos.z)
      controls.target.copy(p)
      camera.position.copy(p).addScaledVector(_rN, -2).addScaledVector(_rB, 0.5)

      // Right-click drag
      let dragging = false, dragStartX = 0, dragStartT = 0, focusDragging = false
      const focusBinding = pane.addBinding(params, 'focusT', { min: 0, max: 1, step: 0.0001, label: 'focus' })
      focusBinding.on('change', () => { if (!focusDragging) updateFocus(rt) })
      canvas.addEventListener('pointerdown', (e) => { if (e.button === 2) { dragging = true; focusDragging = true; dragStartX = e.clientX; dragStartT = params.focusT; e.preventDefault() } })
      canvas.addEventListener('pointermove', (e) => {
        if (!dragging) return
        const totals = getTotalTurns()
        params.focusT = THREE.MathUtils.clamp(dragStartT - (e.clientX - dragStartX) * params.panSpeed / totals[totals.length - 1], 0, 1)
        updateFocus(rt)
      })
      const stopDrag = () => { dragging = false; focusDragging = false }
      canvas.addEventListener('pointerup', stopDrag)
      canvas.addEventListener('pointercancel', stopDrag)
      canvas.addEventListener('contextmenu', (e) => e.preventDefault())

      pane.addBinding(params, 'centuryTurns', { min: 1, max: 20, step: 1, label: 'centuries' })
        .on('change', () => updateFocus(rt))
      pane.addBinding(params, 'coilRadius', { min: 0.1, max: 5, step: 0.1, label: 'coil radius' })
        .on('change', () => updateFocus(rt))
      pane.addBinding(params, 'turnSpacing', { min: 0.1, max: 20, step: 0.1, label: 'turn spacing' })
        .on('change', () => updateFocus(rt))

      const names = ['century', 'decade', 'year', 'day', 'hour', 'min', 'sec']
      for (let i = 1; i < params.offsets.length; i++) {
        const idx = i
        pane.addBinding(params.offsets, idx as unknown as keyof typeof params.offsets, { min: 0.0001, max: i < 3 ? 2 : 0.5, step: 0.0001, label: names[idx] + ' off' })
          .on('change', () => rebuildCoils(rt))
      }
      pane.addBinding(params, 'panSpeed', { min: 1, max: 3600, step: 1, label: 'sec/pixel' })

      const animate = () => {
        if (disposed) return
        rafRef.current = requestAnimationFrame(animate)
        rt.controls.update()
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
      if (rt) { rt.pane.dispose(); rt.controls.dispose(); disposeLevels(rt); rt.lineMaterial.dispose(); rt.renderer.dispose(); runtimeRef.current = null }
    }
  }, [])

  return <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
}
