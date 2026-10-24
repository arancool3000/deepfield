import Foundation
import simd

/// 5.7 Studio finish, the geometry: a scan's depth mesh made ready to sell or print. Floating bits go, every hole is
/// closed by a smooth surface grown across it (rings from its edge, relaxed into a membrane that meets the surface
/// around it), and an object gets a flat closed base - so the model is one watertight piece. Works on indexed triangles.
enum StudioMesh {
    struct Report { var islandsRemoved = 0; var holesFilled = 0; var holesLeft = 0; var addedTris = 0; var watertight = false; var openEdges = 0 }

    /// object: close everything and add a flat base; room: fill holes up to `maxHole` metres round, leave the rest open
    static func finish(v vIn: [SIMD3<Float>], f fIn: [UInt32], object: Bool, h: Float, maxHole: Float = 1.5) -> ([SIMD3<Float>], [UInt32], Report) {
        var rep = Report()
        var (v, f) = keepMain(vIn, fIn, &rep)
        (v, f) = dropBadEdges(v, f)
        let loops = boundaryLoops(f)
        let minY = v.min { $0.y < $1.y }?.y ?? 0
        // the base: the longest loop lying near the bottom (where the floor or table was cut away)
        var baseIdx = -1
        if object {
            var best: Float = 0
            for (i, l) in loops.enumerated() {
                let my = l.map { v[Int($0)].y }.reduce(0, +) / Float(l.count), per = perimeter(v, l)
                if my < minY + max(0.02, h * 6), per > best { best = per; baseIdx = i }
            }
        }
        for (i, l) in loops.enumerated() {
            guard l.count >= 3 else { continue }
            let per = perimeter(v, l)
            if i == baseIdx {
                capFlat(&v, &f, l, y: minY, h: h, &rep)
            } else if object || per <= maxHole {
                fill(&v, &f, l, h: h, &rep)
            } else { rep.holesLeft += 1 }
        }
        let (open, nonManifold) = edgeCheck(f)
        rep.openEdges = open; rep.watertight = open == 0 && nonManifold == 0
        return (v, f, rep)
    }

    /// pieces smaller than 3% of the biggest go (floating specks and stray bits of wall)
    static func keepMain(_ v: [SIMD3<Float>], _ f: [UInt32], _ rep: inout Report) -> ([SIMD3<Float>], [UInt32]) {
        let nv = v.count
        var parent = Array(0..<nv)
        func find(_ a: Int) -> Int { var x = a; while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x] }; return x }
        for t in stride(from: 0, to: f.count, by: 3) {
            let a = find(Int(f[t])), b = find(Int(f[t + 1])), c = find(Int(f[t + 2])); parent[b] = a; parent[find(c)] = a
        }
        var tris = [Int: Int]()
        for t in stride(from: 0, to: f.count, by: 3) { tris[find(Int(f[t])), default: 0] += 1 }
        let big = tris.values.max() ?? 0
        let keep = Set(tris.filter { $0.value * 33 >= big }.keys)
        rep.islandsRemoved = tris.count - keep.count
        var remap = [Int32](repeating: -1, count: nv), ov: [SIMD3<Float>] = [], of: [UInt32] = []
        for t in stride(from: 0, to: f.count, by: 3) where keep.contains(find(Int(f[t]))) {
            for k in 0..<3 {
                let i = Int(f[t + k])
                if remap[i] < 0 { remap[i] = Int32(ov.count); ov.append(v[i]) }
                of.append(UInt32(remap[i]))
            }
        }
        return (ov, of)
    }

    /// triangles on an edge shared by more than two go (they make holes impossible to trace)
    static func dropBadEdges(_ v: [SIMD3<Float>], _ f: [UInt32]) -> ([SIMD3<Float>], [UInt32]) {
        var use = [UInt64: Int]()
        func key(_ a: UInt32, _ b: UInt32) -> UInt64 { UInt64(min(a, b)) << 32 | UInt64(max(a, b)) }
        for t in stride(from: 0, to: f.count, by: 3) { for k in 0..<3 { use[key(f[t + k], f[t + (k + 1) % 3]), default: 0] += 1 } }
        var of: [UInt32] = []; of.reserveCapacity(f.count)
        for t in stride(from: 0, to: f.count, by: 3) {
            let bad = (0..<3).contains { use[key(f[t + $0], f[t + ($0 + 1) % 3])]! > 2 }
            if !bad { of += [f[t], f[t + 1], f[t + 2]] }
        }
        return (v, of)
    }

    /// every hole's edge as a loop of vertices, in the direction the surface's own triangles run along it
    static func boundaryLoops(_ f: [UInt32]) -> [[UInt32]] {
        var dir = Set<UInt64>()
        for t in stride(from: 0, to: f.count, by: 3) { for k in 0..<3 { dir.insert(UInt64(f[t + k]) << 32 | UInt64(f[t + (k + 1) % 3])) } }
        var next = [UInt32: [UInt32]]()
        for e in dir { let a = UInt32(e >> 32), b = UInt32(e & 0xFFFF_FFFF); if !dir.contains(UInt64(b) << 32 | UInt64(a)) { next[a, default: []].append(b) } }
        var loops: [[UInt32]] = []
        while let start = next.keys.first {
            var loop: [UInt32] = [start], cur = start
            while true {
                guard var outs = next[cur], let nx = outs.popLast() else { break }
                if outs.isEmpty { next[cur] = nil } else { next[cur] = outs }
                if nx == start { break }
                // a vertex the loop already passed (two holes touching at a point): split off the inner loop
                if let j = loop.firstIndex(of: nx) { loops.append(Array(loop[j...])); loop.removeSubrange((j + 1)...); cur = nx; continue }
                loop.append(nx); cur = nx
                if loop.count > 2_000_000 { break }
            }
            if loop.count >= 3 { loops.append(loop) }
        }
        return loops
    }

    static func perimeter(_ v: [SIMD3<Float>], _ l: [UInt32]) -> Float {
        var s: Float = 0; for i in 0..<l.count { s += simd_distance(v[Int(l[i])], v[Int(l[(i + 1) % l.count])]) }; return s
    }

    /// a hole closed by rings growing in from its edge, then relaxed so the patch meets the surface around it smoothly
    static func fill(_ v: inout [SIMD3<Float>], _ f: inout [UInt32], _ loop: [UInt32], h: Float, _ rep: inout Report) {
        let n = loop.count
        let c = loop.map { v[Int($0)] }.reduce(SIMD3<Float>.zero, +) / Float(n)
        let radius = loop.map { simd_distance(v[Int($0)], c) }.max() ?? 0
        // rings about 2 cells apart; small holes just get a fan
        let rings = max(0, min(40, Int(radius / (h * 2))))
        let before = f.count
        var prev = loop.map { Int($0) }, newVerts: [Int] = []
        var ringsDone = 0
        for r in 1...max(1, rings) where rings > 0 {
            let t = Float(r) / Float(rings + 1)
            // fewer points on inner rings (keeps triangles even)
            let m = max(3, Int(Float(n) * (1 - t)))
            var ring: [Int] = []
            for k in 0..<m {
                let src = loop[k * n / m]
                v.append(v[Int(src)] * (1 - t) + c * t); ring.append(v.count - 1)
            }
            stitch(&f, outer: prev, inner: ring)
            newVerts += ring; prev = ring; ringsDone = r
        }
        _ = ringsDone
        // centre fan (reversed winding: the patch's triangles run against the hole's edge)
        v.append(c); let ci = v.count - 1; newVerts.append(ci)
        for i in 0..<prev.count { f += [UInt32(prev[(i + 1) % prev.count]), UInt32(prev[i]), UInt32(ci)] }
        // relax: each new point moves to the average of its neighbours (the edge stays put) - a smooth membrane
        if !newVerts.isEmpty { relax(&v, f, from: before, moving: Set(newVerts), passes: 30) }
        rep.holesFilled += 1; rep.addedTris += (f.count - before) / 3
    }

    /// joins two rings with triangles (both listed in the hole edge's direction; winding reversed against it)
    private static func stitch(_ f: inout [UInt32], outer: [Int], inner: [Int]) {
        let a = outer.count, b = inner.count
        var i = 0, j = 0
        while i < a || j < b {
            let oi = outer[i % a], oi1 = outer[(i + 1) % a], ij = inner[j % b], ij1 = inner[(j + 1) % b]
            // advance whichever ring is behind (by share of the way round)
            if j >= b || (i < a && Float(i + 1) / Float(a) <= Float(j + 1) / Float(b)) {
                f += [UInt32(oi1), UInt32(oi), UInt32(ij)]; i += 1
            } else {
                f += [UInt32(ij), UInt32(ij1), UInt32(oi)]; j += 1
            }
            _ = oi1; _ = ij1
        }
    }

    private static func relax(_ v: inout [SIMD3<Float>], _ f: [UInt32], from: Int, moving: Set<Int>, passes: Int) {
        var nb = [Int: Set<Int>]()
        for t in stride(from: from, to: f.count, by: 3) {
            for k in 0..<3 {
                let a = Int(f[t + k]), b = Int(f[t + (k + 1) % 3])
                if moving.contains(a) { nb[a, default: []].insert(b) }
                if moving.contains(b) { nb[b, default: []].insert(a) }
            }
        }
        for _ in 0..<passes {
            var nv = v
            for (i, s) in nb where !s.isEmpty { nv[i] = s.reduce(SIMD3<Float>.zero) { $0 + v[$1] } / Float(s.count) }
            v = nv
        }
    }

    /// the cut-away bottom closed flat: a short wall straight down to the lowest point, then a flat floor
    static func capFlat(_ v: inout [SIMD3<Float>], _ f: inout [UInt32], _ loop: [UInt32], y: Float, h: Float, _ rep: inout Report) {
        let before = f.count
        var down: [Int] = []
        for i in loop { var p = v[Int(i)]; p.y = y; v.append(p); down.append(v.count - 1) }
        let n = loop.count
        for i in 0..<n {
            let a = Int(loop[i]), b = Int(loop[(i + 1) % n]), a2 = down[i], b2 = down[(i + 1) % n]
            f += [UInt32(b), UInt32(a), UInt32(a2), UInt32(b), UInt32(a2), UInt32(b2)]
        }
        var tmp = Report()
        fill(&v, &f, down.map { UInt32($0) }, h: h, &tmp)
        // the floor stays flat: everything the fill added is put back on the base plane
        for i in (before == 0 ? 0 : 0)..<v.count where i >= v.count - (tmp.addedTris + 4) && v[i].y < y + h * 4 { v[i].y = y }
        rep.holesFilled += 1; rep.addedTris += (f.count - before) / 3
    }

    /// open edges (used by one triangle) and edges shared by more than two
    static func edgeCheck(_ f: [UInt32]) -> (Int, Int) {
        var use = [UInt64: Int]()
        for t in stride(from: 0, to: f.count, by: 3) { for k in 0..<3 { let a = f[t + k], b = f[t + (k + 1) % 3]; use[UInt64(min(a, b)) << 32 | UInt64(max(a, b)), default: 0] += 1 } }
        var open = 0, bad = 0
        for (_, c) in use { if c == 1 { open += 1 } else if c > 2 { bad += 1 } }
        return (open, bad)
    }

    /// stands it on y = 0, centred over the origin (real size unchanged)
    static func ground(_ v: inout [SIMD3<Float>]) {
        guard !v.isEmpty else { return }
        var lo = v[0], hi = v[0]; for p in v { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let s = SIMD3<Float>((lo.x + hi.x) / 2, lo.y, (lo.z + hi.z) / 2)
        for i in 0..<v.count { v[i] -= s }
    }
}
