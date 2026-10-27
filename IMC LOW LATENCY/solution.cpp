#include "api.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#ifndef EXTRA_MINN
#define EXTRA_MINN 300.0
#endif
#ifndef EXTRA_Z
#define EXTRA_Z 2.0
#endif
#ifndef EXTRAS
#define EXTRAS 1
#endif
#ifndef PRED
#define PRED 0
#endif
#ifndef USE_CLS
#define USE_CLS 0
#endif
#ifndef USE_SKIP
#define USE_SKIP 1
#endif
#ifndef XK
#define XK 0.8
#endif

// Adaptive pricer.
//
// Pricing: Black-Scholes with everything that does not change precomputed,
// one log(S) per underlying update, fast double-precision normal CDF.
//
// Quoting: a quote is only worth sending if our price is likely close to the
// hidden reference. The main risk is a stale volatility. For every quote we
// record which "situation" it was sent in (how stale its vol was), and the
// profit reported in previousScores is credited to that situation. Each
// option keeps its own table of average profit per staleness bucket, backed
// by a table pooled across all options (bucketed by staleness * vega^2, which
// is what drives the expected price error). We quote only where the estimated
// profit is positive, plus a small optimism bonus so new situations get tried.
//
// Repeat quotes (EXTRAS): every call we also re-quote options the update did not touch, using their
// cached price (an option is scored once per call, so a still-correct price can earn again). These are
// judged by pooled statistics of their own (normalized profit per staleness x vega^2 bucket), cut when
// confidently losing, paced against the time budget, and switched off as a whole if they lose overall.
//
// Vol sources: an option's own last V update. If an option has never had one,
// it borrows the vol of the nearest-strike option on the same underlying; this
// is learned separately, so it is dropped automatically if it loses money.
class Solution : public ISolution {
    static constexpr double R = 0.04;
    static constexpr int NB = 48;   // staleness buckets per vol source
    static constexpr int NSRC = 2;  // 0 = own vol, 1 = borrowed from a sibling
    // Feedback state of an option since its last V: how did the last own-vol quote on this vol score,
    // relative to the option's profit scale m? 0 = no feedback yet, 1 = > 0.66 m, 2 = > 0.33 m, 3 = > 0, 4 = loss.
    // Quotes on updates are learned per (source, state); repeat quotes are skipped after a loss.
    static constexpr int NC = 5;
    static constexpr int NGRP = NSRC * NC;
    static constexpr int NQ = 112;  // pooled buckets
    static constexpr double PRIOR_W = 3.0;  // weight of pooled prior (in samples)
    static constexpr double BONUS = 0.25;   // optimism bonus, in units of option's profit scale
    static constexpr double POOL_OPT = 0.5; // optimistic normalized prior for unseen pooled buckets
    static constexpr double FADE = 0.99;    // old results fade: ~100 most recent quotes per situation count
    static constexpr double CAP_NS = 5.0e8;  // processing time we allow ourselves (budget is 1s)
    static constexpr double ZCUT = 2.0;     // only stop quoting a situation if it is losing by > 2 standard errors
    // A cut situation is still sampled now and then so the gate keeps learning about it:
    // every 4th chance if it is near break-even, every 16th if clearly negative, every 64th if very negative.

    struct Reg {
        int id, u;
        double K, T;
    };

    struct Opt {
        // static
        double K, T, lnK, sqrtT, disc;
        int id, u;
        // own vol (precomputed pricing terms)
        double sig, a, inv, sst;
        long tV;       // step of last own V
        double sigPrev; int hasPrev;   // vol before the last own V
        // cached price for repeat quotes
        double cPrice, cVega;
        unsigned cSV;
        long cTv;
        int cSib, cSrc, cB;
        int own;       // 1 if own vol is known
        int cls;       // feedback state since last V (see NC)
        unsigned lossSkip;  // repeat-quote chances skipped since the last loss
        int sib;       // index of sibling to borrow vol from (-1 = none)
        // profit scale
        double absSum;
        double cnt;
    };

    struct Pend {
        int id, opt, slot, qslot;
        double scale;
        int src;   // 0/1 = quote on update (own / borrowed vol), 2/3 = repeat quote
        int grp;   // pooled-table row: src * NC + state for quotes on updates, NGRP + (src - 2) for repeats
    };

    std::vector<Reg> regs;
    bool built = false;
    long step = 0;

    std::vector<Opt> o;               // grouped by underlying
    std::vector<int> ubeg, uend;      // underlying index -> [beg, end)
    std::vector<double> spot, lnSpot;
    std::vector<char> hasSpot;

    // id lookup: dense arrays when ids are small, hash map otherwise
    bool denseOpt = false, denseU = false;
    std::vector<int> optDense, uDense;
    std::unordered_map<int, int> optMap, uMap;

    // learning tables
    std::vector<float> tN, tS, tQ;    // per option: [opt][src][bucket] count, sum, sum of squares
    std::vector<unsigned> tC;         // per situation: chances skipped since it was cut
    double pN[NGRP + 2][NQ] = {}, pS[NGRP + 2][NQ] = {}, pQ[NGRP + 2][NQ] = {};  // pooled, normalized profits
    unsigned tCq[NGRP + 2][NQ] = {};
    double eS = 0, eS2 = 0, eN = 0;   // all repeat quotes together (normalized)
    std::vector<unsigned> spotVer;
    // learned dynamics of sigma between V updates: sigma_next - sigma = a + b1*(sigma - mean) + b2*(sigma - sigma_prev),
    // fitted per staleness bucket from the V updates themselves (ridge-regularised, zero if there is no structure)
    static constexpr int NRB = 24;
    struct Reg3 { double n = 0, s1 = 0, s2 = 0, sy = 0, s11 = 0, s12 = 0, s22 = 0, s1y = 0, s2y = 0; double al = 0, b1 = 0, b2 = 0; bool dirty = true; } r3[NRB];
    double gMean = 0; int gMeanN = 0;
    std::vector<double> uMean; std::vector<int> uMeanN;
    int curU = -1, curP = -1, rot = 0;
    double usedNs = 0, cpq = 15.0;
    double gAbs = 0, gCnt = 0;        // global |profit| scale

    std::vector<Pend> pend, pendNext;

    static inline double cnd(double x, double& e) {
        double ax = std::fabs(x);
        double c;
        e = std::exp(-0.5 * ax * ax);
        if (ax > 37.0) {
            c = 0.0;
        } else if (ax < 7.07106781186547) {
            double b = 3.52624965998911E-02 * ax + 0.700383064443688;
            b = b * ax + 6.37396220353165;
            b = b * ax + 33.912866078383;
            b = b * ax + 112.079291497871;
            b = b * ax + 221.213596169931;
            b = b * ax + 220.206867912376;
            c = e * b;
            b = 8.83883476483184E-02 * ax + 1.75566716318264;
            b = b * ax + 16.064177579207;
            b = b * ax + 86.7807322029461;
            b = b * ax + 296.564248779674;
            b = b * ax + 637.333633378831;
            b = b * ax + 793.826512519948;
            b = b * ax + 440.413735824752;
            c = c / b;
        } else {
            double b = ax + 0.65;
            b = ax + 4.0 / b;
            b = ax + 3.0 / b;
            b = ax + 2.0 / b;
            b = ax + 1.0 / b;
            c = e / b / 2.506628274631;
        }
        return x > 0 ? 1.0 - c : c;
    }

    static inline int log2b(unsigned v) { return v ? 32 - __builtin_clz(v) : 0; }

    // Bucket of staleness s (global steps since the vol was set): 0 for s = 0,
    // then two buckets per doubling.
    static inline int sBucket(long s) {
        if (s <= 0) return 0;
        int lz = 63 - __builtin_clzll((unsigned long long)s);
        int half = (double)s >= 1.41421356 * (double)(1ULL << lz);
        int b = 1 + 2 * lz + half;
        return b < NB ? b : NB - 1;
    }

    // Pooled bucket of s * vega^2 (expected squared price error up to a constant).
    static inline int qBucket(long s, double vega) {
        if (s <= 0) return 0;
        double x = (double)s * vega * vega;
        if (!(x > 0)) return 1;
        int b = 60 + (int)std::floor(2.0 * std::log2(x));
        if (b < 1) b = 1;
        if (b >= NQ) b = NQ - 1;
        return b;
    }

    int findOpt(int id) const {
        if (denseOpt) return (id >= 0 && id < (int)optDense.size()) ? optDense[id] : -1;
        auto it = optMap.find(id);
        return it == optMap.end() ? -1 : it->second;
    }
    int findU(int id) const {
        if (denseU) return (id >= 0 && id < (int)uDense.size()) ? uDense[id] : -1;
        auto it = uMap.find(id);
        return it == uMap.end() ? -1 : it->second;
    }

    void build() {
        built = true;
        // map underlying ids to 0..nU-1 in order of first appearance
        std::unordered_map<int, int> um;
        std::vector<int> uOrig;
        for (const Reg& r : regs)
            if (um.find(r.u) == um.end()) {
                um[r.u] = (int)uOrig.size();
                uOrig.push_back(r.u);
            }
        int nU = (int)uOrig.size(), n = (int)regs.size();
        std::vector<int> cnt(nU + 1, 0);
        for (const Reg& r : regs) cnt[um[r.u] + 1]++;
        for (int i = 0; i < nU; i++) cnt[i + 1] += cnt[i];
        ubeg.assign(cnt.begin(), cnt.end() - 1);
        uend.assign(cnt.begin() + 1, cnt.end());

        o.assign(n, Opt{});
        std::vector<int> fill(ubeg);
        std::vector<int> pos(n);
        for (int k = 0; k < n; k++) {
            const Reg& r = regs[k];
            int u = um[r.u];
            int p = fill[u]++;
            pos[k] = p;
            Opt& x = o[p];
            x.K = r.K;
            x.T = r.T;
            x.lnK = r.K > 0 ? std::log(r.K) : 0;
            x.sqrtT = r.T > 0 ? std::sqrt(r.T) : 0;
            x.disc = r.K * std::exp(-R * (r.T > 0 ? r.T : 0));
            x.id = r.id;
            x.u = u;
            x.own = 0;
            x.sib = -1;
            x.tV = 0;
        }

        // id lookups
        long minO = 0, maxO = -1, minU = 0, maxU = -1;
        for (const Reg& r : regs) {
            minO = std::min<long>(minO, r.id); maxO = std::max<long>(maxO, r.id);
            minU = std::min<long>(minU, r.u);  maxU = std::max<long>(maxU, r.u);
        }
        denseOpt = minO >= 0 && maxO < 8L * n + 4096;
        denseU = minU >= 0 && maxU < 8L * nU + 4096;
        if (denseOpt) {
            optDense.assign(maxO + 1, -1);
            for (int k = 0; k < n; k++) optDense[regs[k].id] = pos[k];
        } else {
            optMap.reserve(2 * n);
            for (int k = 0; k < n; k++) optMap[regs[k].id] = pos[k];
        }
        if (denseU) {
            uDense.assign(maxU + 1, -1);
            for (int u = 0; u < nU; u++) uDense[uOrig[u]] = u;
        } else {
            uMap = um;
        }

        spot.assign(nU, 0);
        lnSpot.assign(nU, 0);
        hasSpot.assign(nU, 0);
        spotVer.assign(nU, 0u);
        uMean.assign(nU, 0.0); uMeanN.assign(nU, 0);
        tN.assign((size_t)n * NGRP * NB, 0.f);
        tS.assign((size_t)n * NGRP * NB, 0.f);
        tQ.assign((size_t)n * NGRP * NB, 0.f);
        tC.assign((size_t)n * NGRP * NB, 0u);
        pend.reserve(n + 8);
        pendNext.reserve(n + 8);
        regs.clear();
        regs.shrink_to_fit();
    }

    // Credit last call's profits to the situations the quotes were sent in.
    void learn(const std::vector<Score>& sc) {
        size_t j = 0;
        for (const Score& s : sc) {
            while (j < pend.size() && pend[j].id != s.optionId) j++;
            if (j == pend.size()) break;
            const Pend& p = pend[j++];
            double pr = s.profit;
            if (!std::isfinite(pr)) continue;
            Opt& x = o[p.opt];
            double a = std::fabs(pr);
            x.absSum += a;
            x.cnt += 1;
            gAbs += a;
            gCnt += 1;
            if (p.slot >= 0) {
                tN[p.slot] = tN[p.slot] * (float)FADE + 1.f;
                tS[p.slot] = tS[p.slot] * (float)FADE + (float)pr;
                tQ[p.slot] = tQ[p.slot] * (float)FADE + (float)(pr * pr);
            }
            int src = p.src, g = p.grp;
            double z = pr / p.scale;
            pN[g][p.qslot] = pN[g][p.qslot] * FADE + 1;
            pS[g][p.qslot] = pS[g][p.qslot] * FADE + z;
            pQ[g][p.qslot] = pQ[g][p.qslot] * FADE + z * z;
            if (src == 0 || src == 2) x.cls = z > 0.66 ? 1 : (z > 0.33 ? 2 : (z > 0 ? 3 : 4));
            if (src >= 2) {
                eS = eS * 0.9995 + z; eS2 = eS2 * 0.9995 + z * z; eN = eN * 0.9995 + 1.0;
            }
        }
    }

    inline double scaleOf(const Opt& x) const {
        double g = gCnt > 0 ? gAbs / gCnt : 1.0;
        if (!(g > 0)) g = 1.0;
        double m = (x.absSum + 2.0 * g) / (x.cnt + 2.0);
        return m > 1e-12 ? m : 1e-12;
    }

    // Price option p with spot S and (a, inv, sst), decide whether to quote.
    inline void consider(int p, int src, long tV, double sig, double a, double inv, double sst,
                         double S, double lnS, std::vector<Valuation>& out) {
        const Opt& x = o[p];
        double price, vega;
        if (x.T > 0 && x.K > 0 && sig > 0) {
            double e1, e2;
            double d1 = (lnS + a) * inv;
            double n1 = cnd(d1, e1);
            double n2 = cnd(d1 - sst, e2);
            price = S * n1 - x.disc * n2;
            vega = S * x.sqrtT * e1 * 0.3989422804014327;
        } else {
            price = S - x.disc;  // T <= 0, K <= 0 or zero vol: intrinsic value
            if (price < 0) price = 0;
            vega = 0;
        }
        if (!std::isfinite(price)) return;

        long s = step - tV;
        int b = sBucket(s);
        int q = qBucket(s, vega);
        int g = src * NC + ((USE_CLS && src == 0) ? x.cls : 0);
        size_t slot = ((size_t)p * NGRP + g) * NB + b;
        double m = scaleOf(x);
        double pm = (pS[g][q] + 2.0 * POOL_OPT) / (pN[g][q] + 2.0);
        double n = tN[slot];
        double est = (tS[slot] + PRIOR_W * pm * m) / (n + PRIOR_W);
        // spread of profits in this situation (prior spread = profit scale m)
        double var = (tQ[slot] + PRIOR_W * m * m) / (n + PRIOR_W) - est * est;
        double se = std::sqrt(std::max(var, 0.0) / (n + PRIOR_W));
        if (est + ZCUT * se + BONUS * m / std::sqrt(n + 1.0) <= 0) {
            double z = est / m;
            unsigned every = z > -0.25 ? 4u : (z > -1.0 ? 16u : 64u);
            if (++tC[slot] % every != 0) return;  // skip, but sample occasionally
        }

        out.push_back(Valuation{x.id, price});
        pendNext.push_back(Pend{x.id, p, (int)slot, q, m, src, g});
    }

    inline void calc(const Opt& x, double sig, double a, double inv, double sst,
                     double S, double lnS, double& price, double& vega) const {
        if (x.T > 0 && x.K > 0 && sig > 0) {
            double e1, e2;
            double d1 = (lnS + a) * inv;
            double n1 = cnd(d1, e1);
            double n2 = cnd(d1 - sst, e2);
            price = S * n1 - x.disc * n2;
            vega = S * x.sqrtT * e1 * 0.3989422804014327;
        } else {
            price = S - x.disc;
            if (price < 0) price = 0;
            vega = 0;
        }
    }

    // Repeat quote of an option this update did not touch (cached price, pooled gating).
    inline void considerExtra(int p, std::vector<Valuation>& out) {
        Opt& x = o[p];
        int u = x.u;
        if (!hasSpot[u]) return;
        int src, sib = -1;
        long tV;
        double sig, a, inv, sst;
        int cb = 0;
        if (x.own) {
            src = 0; tV = x.tV;
#if PRED
            long ss = step - x.tV;
            cb = ss <= 0 ? 0 : 1 + log2b((unsigned)std::min<long>(ss, 1L << 30));
            sig = adjSigma(x, ss);
            sst = sig * x.sqrtT;
            inv = sst > 0 ? 1.0 / sst : 0;
            a = (R + 0.5 * sig * sig) * x.T - x.lnK;
#else
            sig = x.sig; a = x.a; inv = x.inv; sst = x.sst;
#endif
        } else if (x.sib >= 0) {
            const Opt& y = o[x.sib];
            src = 1; sib = x.sib; tV = y.tV; sig = y.sig;
            sst = sig * x.sqrtT;
            inv = sst > 0 ? 1.0 / sst : 0;
            a = (R + 0.5 * sig * sig) * x.T - x.lnK;
        } else {
            return;
        }
        if (x.cSV != spotVer[u] || x.cTv != tV || x.cSib != sib || x.cSrc != src || x.cB != cb) {
            double pr, vg;
            calc(x, sig, a, inv, sst, spot[u], lnSpot[u], pr, vg);
            x.cPrice = pr; x.cVega = vg; x.cSV = spotVer[u]; x.cTv = tV; x.cSib = sib; x.cSrc = src; x.cB = cb;
        }
        double price = x.cPrice;
        if (!std::isfinite(price)) return;
        // last own-vol quote on this vol lost: the vol has drifted, so do not repeat (sample 1 in 16)
        if (USE_SKIP && src == 0 && x.cls == 4 && (++x.lossSkip & 15) != 0) return;
        long s = step - tV;
        int q = qBucket(s, x.cVega);
        int es = src + 2;
        int g = NGRP + src;
        double n = pN[g][q];
        double mean = (pS[g][q] + 2.0 * POOL_OPT) / (n + 2.0);
        double var = (pQ[g][q] + 2.0 * (POOL_OPT * POOL_OPT + 1.0)) / (n + 2.0) - mean * mean;
        double se = std::sqrt(std::max(var, 0.0) / (n + 2.0));
        if (mean - XK * se <= 0) {
            // not (yet) shown to be profitable: only a sample of the chances are taken
            unsigned every = n < 20.0 ? 3u : (mean > -0.25 ? 8u : (mean > -1.0 ? 32u : 128u));
            if (++tCq[g][q] % every != 0) return;
        }
        double m = scaleOf(x);
        out.push_back(Valuation{x.id, price});
        pendNext.push_back(Pend{x.id, p, -1, q, m, es, g});
    }

    inline double meanFor(int u) const { return uMeanN[u] >= 3 ? uMean[u] : gMean; }

    void solve(Reg3& r) {
        r.dirty = false;
        r.al = r.b1 = r.b2 = 0;
        if (r.n < 30.0) return;
        double l1 = 2.0 * r.s11 / (r.n + 1.0) + 1e-12, l2 = 2.0 * r.s22 / (r.n + 1.0) + 1e-12;
        double A[3][4] = {{r.n, r.s1, r.s2, r.sy}, {r.s1, r.s11 + l1, r.s12, r.s1y}, {r.s2, r.s12, r.s22 + l2, r.s2y}};
        for (int c = 0; c < 3; c++) {
            int pv = c;
            for (int i = c + 1; i < 3; i++) if (std::fabs(A[i][c]) > std::fabs(A[pv][c])) pv = i;
            if (std::fabs(A[pv][c]) < 1e-14) return;
            if (pv != c) for (int j = 0; j < 4; j++) std::swap(A[pv][j], A[c][j]);
            for (int i = c + 1; i < 3; i++) {
                double f = A[i][c] / A[c][c];
                for (int j = c; j < 4; j++) A[i][j] -= f * A[c][j];
            }
        }
        double x[3];
        for (int i = 2; i >= 0; i--) {
            double v = A[i][3];
            for (int j = i + 1; j < 3; j++) v -= A[i][j] * x[j];
            x[i] = v / A[i][i];
        }
        if (!std::isfinite(x[0] + x[1] + x[2])) return;
        r.al = x[0]; r.b1 = x[1]; r.b2 = x[2];
    }

    // Best guess of the current vol of an option with own vol, given how long ago it was reported.
    inline double adjSigma(const Opt& x, long s) {
#if PRED
        if (s <= 0 || !x.hasPrev) return x.sig;
        Reg3& r = r3[std::min(NRB - 1, (int)log2b((unsigned)std::min<long>(s, 1L << 30)))];
        if (r.n < 30.0) return x.sig;
        if (r.dirty) solve(r);
        double d = r.al + r.b1 * (x.sig - meanFor(x.u)) + r.b2 * (x.sig - x.sigPrev);
        double lim = 0.5 * x.sig;
        if (d > lim) d = lim;
        if (d < -lim) d = -lim;
        double v = x.sig + d;
        return v < 0.01 ? 0.01 : v;
#else
        (void)s;
        return x.sig;
#endif
    }

    inline void quoteOption(int p, double S, double lnS, std::vector<Valuation>& out) {
        const Opt& x = o[p];
        if (x.own) {
            double sg = adjSigma(x, step - x.tV);
            if (sg == x.sig) {
                consider(p, 0, x.tV, x.sig, x.a, x.inv, x.sst, S, lnS, out);
            } else {
                double sst = sg * x.sqrtT;
                double inv = sst > 0 ? 1.0 / sst : 0;
                double a = (R + 0.5 * sg * sg) * x.T - x.lnK;
                consider(p, 0, x.tV, sg, a, inv, sst, S, lnS, out);
            }
        } else if (x.sib >= 0) {
            const Opt& y = o[x.sib];
            double sig = y.sig;
            double sst = sig * x.sqrtT;
            double inv = sst > 0 ? 1.0 / sst : 0;
            double a = (R + 0.5 * sig * sig) * x.T - x.lnK;
            consider(p, 1, y.tV, sig, a, inv, sst, S, lnS, out);
        }
    }

    void setVol(int p, double sig) {
        Opt& x = o[p];
#if PRED
        if (x.own && x.hasPrev && step > x.tV) {
            double m = meanFor(x.u);
            double x1 = x.sig - m, x2 = x.sig - x.sigPrev, y = sig - x.sig;
            Reg3& r = r3[std::min(NRB - 1, (int)log2b((unsigned)std::min<long>(step - x.tV, 1L << 30)))];
            const double D = 0.998;
            r.n = r.n * D + 1; r.s1 = r.s1 * D + x1; r.s2 = r.s2 * D + x2; r.sy = r.sy * D + y;
            r.s11 = r.s11 * D + x1 * x1; r.s12 = r.s12 * D + x1 * x2; r.s22 = r.s22 * D + x2 * x2;
            r.s1y = r.s1y * D + x1 * y; r.s2y = r.s2y * D + x2 * y;
            r.dirty = true;
        }
        if (x.own) { x.sigPrev = x.sig; x.hasPrev = 1; }
        {   // running means of reported vols (per underlying and overall)
            gMean = gMeanN == 0 ? sig : gMean + 0.02 * (sig - gMean); gMeanN++;
            int u = x.u;
            uMean[u] = uMeanN[u] == 0 ? sig : uMean[u] + 0.05 * (sig - uMean[u]); uMeanN[u]++;
        }
#endif
        x.sig = sig;
        x.sst = sig * x.sqrtT;
        x.inv = x.sst > 0 ? 1.0 / x.sst : 0;
        x.a = (R + 0.5 * sig * sig) * x.T - x.lnK;
        x.tV = step;
        x.cls = 0;
        bool wasOwn = x.own;
        x.own = 1;
        if (!wasOwn) {
            // options on this underlying without their own vol borrow from the nearest strike
            for (int q = ubeg[x.u]; q < uend[x.u]; q++) {
                Opt& y = o[q];
                if (y.own) continue;
                if (y.sib < 0 || std::fabs(y.K - x.K) < std::fabs(y.K - o[y.sib].K)) y.sib = p;
            }
        }
    }

public:
    void registerOption(const Option& option) override {
        regs.push_back(Reg{option.id, option.underlyingId, (double)option.strike, option.tte});
    }

    std::vector<Valuation> process(const ParameterUpdate& update,
                                   const std::vector<Score>& previousScores) override {
        bool timed = ((step & 7) == 7);
        std::chrono::steady_clock::time_point t0;
        if (timed) t0 = std::chrono::steady_clock::now();
        if (!built) build();
        if (!previousScores.empty()) learn(previousScores);
        pendNext.clear();
        std::vector<Valuation> out;
        curU = -1; curP = -1;

        if (update.type == 'U') {
            int u = findU(update.instrumentId);
            double S = update.value;
            if (u >= 0) {
                if (S > 0 && std::isfinite(S)) {
                    double lnS = std::log(S);
                    spot[u] = S;
                    spotVer[u]++;
                    curU = u;
                    lnSpot[u] = lnS;
                    hasSpot[u] = 1;
                    out.reserve(uend[u] - ubeg[u]);
                    for (int p = ubeg[u]; p < uend[u]; p++) quoteOption(p, S, lnS, out);
                } else {
                    hasSpot[u] = 0;
                }
            }
        } else if (update.type == 'V') {
            int p = findOpt(update.instrumentId);
            double sig = update.value;
            if (p >= 0 && sig >= 0 && std::isfinite(sig)) {
                setVol(p, sig);
                curP = p;
                int u = o[p].u;
                if (hasSpot[u]) quoteOption(p, spot[u], lnSpot[u], out);
            }
        }
#if EXTRAS
        {
            int n = (int)o.size();
            bool on = true;
            if (eN >= EXTRA_MINN) {   // all repeat quotes together confidently losing -> mostly off
                double mm = eS / eN;
                double vv = std::max(0.0, eS2 / eN - mm * mm);
                if (mm + EXTRA_Z * std::sqrt(vv / eN) < 0.0) on = ((step & 15) == 0);
            }
            if (on && n > 0) {
                double allowed = (CAP_NS - usedNs) / ((double)step + 1000.0);
                long budgetQ = allowed > 0 ? (long)(allowed / cpq) : 0;
                long extraK = std::min<long>(n, std::max<long>(0, budgetQ - (long)out.size()));
                for (long j = 0; j < extraK; j++) {
                    int idx = rot;
                    if (++rot >= n) rot = 0;
                    if (o[idx].u == curU || idx == curP) continue;
                    considerExtra(idx, out);
                }
            }
        }
#endif
        pend.swap(pendNext);
        step++;
        if (timed) {
            double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
            if (ns > 2.0e6) ns = 2.0e6;
            usedNs += ns * 8.0;
            if (!out.empty()) {
                double sp = ns / (double)out.size();
                cpq = 0.9 * cpq + 0.1 * std::min(std::max(sp, 2.0), 400.0);
            }
        }
        return out;
    }
};

ISolution* createSolution() {
    return new Solution();
}