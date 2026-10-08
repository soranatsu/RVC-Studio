#include "rubberband/RubberBandLiveShifter.h"
#include "rubberband/RubberBandStretcher.h"
#include <cmath>
extern "C" {
struct rb_handle { RubberBand::RubberBandLiveShifter *p; size_t channels, block; };
__declspec(dllexport) rb_handle *rb_create(int sr, int ch, double ratio) {
    if (sr <= 0 || ch <= 0 || !std::isfinite(ratio) || ratio <= 0) return nullptr;
    auto *h = new rb_handle{new RubberBand::RubberBandLiveShifter(sr, ch,
        RubberBand::RubberBandLiveShifter::OptionFormantPreserved), (size_t)ch, 0};
    h->p->setPitchScale(ratio); h->block = h->p->getBlockSize(); return h;
}

extern "C" {
struct rb_r3 { RubberBand::RubberBandStretcher *p; size_t block; };
__declspec(dllexport) rb_r3 *rb_r3_create_mode(int sr, int ch, double ratio, int finer) {
    auto opts = RubberBand::RubberBandStretcher::OptionProcessRealTime |
                (finer ? RubberBand::RubberBandStretcher::OptionEngineFiner : RubberBand::RubberBandStretcher::OptionEngineFaster) |
                RubberBand::RubberBandStretcher::OptionWindowShort |
                RubberBand::RubberBandStretcher::OptionFormantPreserved |
                RubberBand::RubberBandStretcher::OptionPitchHighConsistency;
    auto *p = new RubberBand::RubberBandStretcher(sr, ch, opts, 1.0, ratio);
    p->setMaxProcessSize(4096);
    return new rb_r3{p, 4096};
}
__declspec(dllexport) rb_r3 *rb_r3_create(int sr, int ch, double ratio) { return rb_r3_create_mode(sr,ch,ratio,1); }
__declspec(dllexport) size_t rb_r3_delay(rb_r3 *h) { return h ? h->p->getStartDelay() : 0; }
__declspec(dllexport) size_t rb_r3_pad(rb_r3 *h) { return h ? h->p->getPreferredStartPad() : 0; }
__declspec(dllexport) void rb_r3_process(rb_r3 *h, const float *const *in, size_t n, int final) { if (h) h->p->process(in,n,final != 0); }
__declspec(dllexport) int rb_r3_available(rb_r3 *h) { return h ? h->p->available() : -1; }
__declspec(dllexport) size_t rb_r3_retrieve(rb_r3 *h, float *const *out, size_t n) { return h ? h->p->retrieve(out,n) : 0; }
__declspec(dllexport) void rb_r3_delete(rb_r3 *h) { if (h) { delete h->p; delete h; } }
}
__declspec(dllexport) void rb_set_pitch(rb_handle *h, double ratio) { if (h) h->p->setPitchScale(ratio); }
__declspec(dllexport) size_t rb_block_size(rb_handle *h) { return h ? h->block : 0; }
__declspec(dllexport) size_t rb_start_delay(rb_handle *h) { return h ? h->p->getStartDelay() : 0; }
__declspec(dllexport) int rb_shift(rb_handle *h, const float *const *in, float *const *out) { if (!h) return 0; h->p->shift(in, out); return 1; }
__declspec(dllexport) void rb_delete(rb_handle *h) { if (h) { delete h->p; delete h; } }
}
