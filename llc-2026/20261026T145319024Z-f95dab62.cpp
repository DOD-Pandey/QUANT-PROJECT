#include "api.hpp"

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

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
// Vol sources: an option's own last V update. If an option has never had one,
// it borrows the vol of the nearest-strike option on the same underlying; this
// is learned separately, so it is dropped automatically if it loses money.
class Solution : public ISolution {
    static constexpr double R = 0.04;
    static constexpr int NB = 48;   // staleness buckets per vol source
    static constexpr int NSRC = 2;  // 0 = own vol, 1 = borrowed from a sibling
    static constexpr int NQ = 112;  // pooled buckets
    static constexpr double PRIOR_W = 3.0;  // weight of pooled prior (in samples)
    static constexpr double BONUS = 0.25;   // optimism bonus, in units of option's profit scale
    static constexpr double POOL_OPT = 0.5; // optimistic normalized prior for unseen pooled buckets

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
        int own;       // 1 if own vol is known
        int sib;       // index of sibling to borrow vol from (-1 = none)
        // profit scale
        double absSum;
        double cnt;
    };

    struct Pend {
        int id, opt, slot, qslot;
        double scale;
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
    std::vector<float> tN, tS;        // per option: [opt][src][bucket]
    double pN[NSRC][NQ] = {}, pS[NSRC][NQ] = {};  // pooled, normalized profits
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
        tN.assign((size_t)n * NSRC * NB, 0.f);
        tS.assign((size_t)n * NSRC * NB, 0.f);
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
            tN[p.slot] += 1.f;
            tS[p.slot] += (float)pr;
            int src = p.slot / NB % NSRC;
            pN[src][p.qslot] += 1;
            pS[src][p.qslot] += pr / p.scale;
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
        size_t slot = ((size_t)p * NSRC + src) * NB + b;
        double m = scaleOf(x);
        double pm = (pS[src][q] + 2.0 * POOL_OPT) / (pN[src][q] + 2.0);
        double n = tN[slot];
        double est = (tS[slot] + PRIOR_W * pm * m) / (n + PRIOR_W);
        if (est + BONUS * m / std::sqrt(n + 1.0) <= 0) return;

        out.push_back(Valuation{x.id, price});
        pendNext.push_back(Pend{x.id, p, (int)slot, q, m});
    }

    inline void quoteOption(int p, double S, double lnS, std::vector<Valuation>& out) {
        const Opt& x = o[p];
        if (x.own) {
            consider(p, 0, x.tV, x.sig, x.a, x.inv, x.sst, S, lnS, out);
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
        x.sig = sig;
        x.sst = sig * x.sqrtT;
        x.inv = x.sst > 0 ? 1.0 / x.sst : 0;
        x.a = (R + 0.5 * sig * sig) * x.T - x.lnK;
        x.tV = step;
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
        if (!built) build();
        if (!previousScores.empty()) learn(previousScores);
        pendNext.clear();
        std::vector<Valuation> out;

        if (update.type == 'U') {
            int u = findU(update.instrumentId);
            double S = update.value;
            if (u >= 0) {
                if (S > 0 && std::isfinite(S)) {
                    double lnS = std::log(S);
                    spot[u] = S;
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
                int u = o[p].u;
                if (hasSpot[u]) quoteOption(p, spot[u], lnSpot[u], out);
            }
        }
        pend.swap(pendNext);
        step++;
        return out;
    }
};

ISolution* createSolution() {
    return new Solution();
}