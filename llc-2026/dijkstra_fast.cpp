// Fast Dijkstra: single-source shortest paths, non-negative weights.
//
// Input  (stdin):  n m s
//                  u v w     (m lines, 0-indexed nodes, directed edges)
// Output (stdout): n lines, dist from s to each node (-1 if unreachable)
//
// Assumes every shortest distance fits in 32 bits (< 4,294,967,295).
// If not, switch dist to uint64_t and use a pair-based heap instead of packing.
//
// Build: g++ -O3 -march=native -DNDEBUG -flto dijkstra_fast.cpp -o dijkstra
// Bench: add -DBENCH to print timing to stderr.

#pragma GCC optimize("O3,unroll-loops")
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>
#ifdef BENCH
#include <chrono>
#endif

using u32 = uint32_t;
using u64 = uint64_t;

static constexpr u32 INF = 0xFFFFFFFFu;

// ---------- Fast input: read everything once, parse by hand ----------
static char* ibuf;
static size_t ilen, ipos;

static void read_all() {
    size_t cap = 1 << 20;
    ibuf = (char*)malloc(cap);
    ilen = 0;
    size_t got;
    while ((got = fread(ibuf + ilen, 1, cap - ilen, stdin)) > 0) {
        ilen += got;
        if (ilen == cap) {
            cap <<= 1;
            ibuf = (char*)realloc(ibuf, cap);
        }
    }
}

static inline u32 read_u32() {
    while (ipos < ilen && (ibuf[ipos] < '0' || ibuf[ipos] > '9')) ++ipos;
    u32 x = 0;
    while (ipos < ilen && ibuf[ipos] >= '0' && ibuf[ipos] <= '9')
        x = x * 10 + (ibuf[ipos++] - '0');
    return x;
}

// ---------- Graph in CSR form: contiguous, cache friendly ----------
struct Edge {
    u32 to, w;  // interleaved so one cache line load gets both
};

int main() {
    read_all();
    u32 n = read_u32(), m = read_u32(), s = read_u32();

    // Read edges once, then counting-sort them into CSR (no per-node vectors)
    std::vector<u32> eu(m), ev(m), ew(m);
    std::vector<u32> off(n + 2, 0);
    for (u32 i = 0; i < m; ++i) {
        eu[i] = read_u32();
        ev[i] = read_u32();
        ew[i] = read_u32();
        ++off[eu[i] + 1];
    }
    for (u32 i = 0; i < n; ++i) off[i + 1] += off[i];

    std::vector<Edge> adj(m);
    {
        std::vector<u32> pos(off.begin(), off.begin() + n);
        for (u32 i = 0; i < m; ++i) adj[pos[eu[i]]++] = {ev[i], ew[i]};
    }

#ifdef BENCH
    auto t0 = std::chrono::steady_clock::now();
#endif

    // ---------- Dijkstra ----------
    std::vector<u32> dist(n, INF);
    // Heap entries packed as (dist << 32 | node): one 64-bit compare, 8 bytes each.
    std::vector<u64> heap;
    heap.reserve((size_t)m + 2);  // no reallocation on the hot path
    std::greater<u64> cmp;

    dist[s] = 0;
    heap.push_back((u64)s);

    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        u64 top = heap.back();
        heap.pop_back();
        u32 d = (u32)(top >> 32);
        u32 u = (u32)top;
        if (d != dist[u]) continue;  // stale entry (lazy deletion)

        const Edge* e = adj.data() + off[u];
        const Edge* end = adj.data() + off[u + 1];
        for (; e != end; ++e) {
            u32 nd = d + e->w;
            if (nd < dist[e->to]) {
                dist[e->to] = nd;
                heap.push_back(((u64)nd << 32) | e->to);
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
        }
    }

#ifdef BENCH
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "dijkstra: %.3f ms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count());
#endif

    // ---------- Fast output: build in one buffer, write once ----------
    std::vector<char> out((size_t)n * 12 + 16);
    char* p = out.data();
    for (u32 i = 0; i < n; ++i) {
        if (dist[i] == INF) {
            *p++ = '-';
            *p++ = '1';
        } else {
            char tmp[12];
            int len = 0;
            u32 x = dist[i];
            do {
                tmp[len++] = '0' + x % 10;
                x /= 10;
            } while (x);
            while (len) *p++ = tmp[--len];
        }
        *p++ = '\n';
    }
    fwrite(out.data(), 1, p - out.data(), stdout);
    return 0;
}
